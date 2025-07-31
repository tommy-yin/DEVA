import argparse
import json
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import glob

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def setup_model(model_id):
    model = AutoModel.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto",
        trust_remote_code=True
    ).eval()
    processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, processor

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    pixel_values = transform(image).unsqueeze(0)  # (1, C, H, W)
    return pixel_values

def caption_internvl3(image_path, model, processor):
    pixel_values = load_image(image_path).to(torch.bfloat16).to(model.device)
    prompt = "Please provide a detailed description of the content of the video frame. NOTE: Ignore text and timestamp on the frame. Focus solely on the actual content and activities occurring in the video."
    messages = [{"role": "user", "content": f"[IMAGE] [TEXT] {prompt}"}]
    generation_config = dict(max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9, top_k=50)
    response = model.chat(processor, pixel_values, messages[0]['content'], generation_config)
    return response

def questions_internvl3(prompt, model, processor):
    # 用InternVL3-8B生成questions
    messages = [{"role": "user", "content": prompt}]
    generation_config = dict(max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.9, top_k=50)
    # dummy image for text-only prompt
    dummy_image = torch.zeros((1, 3, 448, 448), dtype=torch.bfloat16).to(model.device)
    response = model.chat(processor, dummy_image, messages[0]['content'], generation_config)
    return response

def extract_subquestions(text):
    import re

     # 提取所有 Guiding Questions
    guiding = re.findall(r"Guiding Questions:\s*(.+)", text)
    # 提取所有 Sub-question
    subqs = re.findall(r"- Sub-question \d+:\s*(.+)", text)
    return guiding, subqs

def build_prompt(description: str, guiding_questions: dict) -> str:
    guiding_json_str = json.dumps(guiding_questions, ensure_ascii=False, indent=2)
    
    # 原版本
    prompt_template = f"""
        You are a surveillance analysis expert.

        Given:
        1. A video frame.
        2. A descriptions of video frame.
        3. A list of Guiding Questions.


        Your task:
        - Pick the top 3 most relevant Guiding Questions based on the Description.  
        - For each selected Guiding Question, break it down into 2-3 detailed sub-questions, making each sub-question specific and grounded based on this frame content:

        ## Input:

        Description:
        {description}

        Guiding Questions:
        {guiding_json_str}

        ## Output format:

        Guiding Questions ID: {{id}}  
        Guiding Questions: {{text}}  
        - Sub-question 1:  
        - Sub-question 2:  
        - Sub-question 3:  
        ...
        (Repeat for three Guiding Questions)
        """
    
    # # 新版本 task描述修改
    # prompt_template = f"""
    #     You are a surveillance analysis expert.

    #     Given:
    #     1. A video frame.
    #     2. A descriptions of video frame.
    #     3. A list of Guiding Questions.


    #     Your task:
    #     - From the full set of Guiding Questions, select the three that best align with the visual details of this frame.  
    #     - For each chosen Guiding Question, generate 1-3 focused sub-questions by rephrasing and grounding the original question in the specific objects, actions, and context in this frame.


    #     ## Input:

    #     Description:
    #     {description}

    #     Guiding Questions:
    #     {guiding_json_str}

    #     ## Output format:

    #     Guiding Questions ID: {{id}}  
    #     Guiding Questions: {{text}}  
    #     - Sub-question 1:  
    #     - Sub-question 2:  
    #     - Sub-question 3:  
    #     ...
    #     (Repeat for three Guiding Questions)
    #     """

    return prompt_template

def get_image_paths(root_folder):
    sample_json = []
    sample_json.extend(glob.glob(os.path.join(root_folder, '*.json')))

    sample_frames = []
    for json_file in sample_json:
        with open(json_file, 'r') as f:
            data = json.load(f)
            frames = data['sample_frames']
            sample_frames.extend(frames)
    return sample_frames


def main(args):
    # 1. 加载模型
    model, processor = setup_model(args.model_id)

    # 2. 读取指导问题 JSON
    with open(args.guiding_questions_json, "r", encoding="utf-8") as f_gq:
        guiding_data = json.load(f_gq)
    guiding_questions = guiding_data.get("round_0001", {})

    # 3.    
    results = {}
    frames = get_image_paths(args.data_root)
    print(f"获取到 {len(frames)} 个帧进行处理")
    
    # 按视频分组处理结果
    for img_path in tqdm(frames, desc="Processing frames"):
        video_folder = os.path.dirname(img_path)
        frame_key = os.path.basename(img_path)
        if video_folder not in results:
            results[video_folder] = {}
        # 3.1 Caption轮
        pixel_values = load_image(img_path).to(torch.bfloat16).to(model.device)
        caption_prompt = "Please provide a detailed description of the content of the video frame. NOTE: Ignore text and timestamp on the frame. Focus solely on the actual content and activities occurring in the video."
        caption_input = f"<image>\n{caption_prompt}"
        generation_config = dict(max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9, top_k=50)
        description, history = model.chat(
            processor, pixel_values, caption_input, generation_config, return_history=True
        )
        results[video_folder][frame_key] = {"description": description}
        print("description:", description)
        # 3.2 Question Generation轮（带history）
        prompt = build_prompt(description, guiding_questions)
        questions_text, history = model.chat(
            processor, pixel_values, prompt, generation_config, history=history, return_history=True
        )
        guiding, subqs = extract_subquestions(questions_text)
        results[video_folder][frame_key]["original_questions"] = guiding
        results[video_folder][frame_key]["questions"] = subqs
        print("original_questions:", guiding)
        print("questions:", subqs)

        with open(args.descriptions_json, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, indent=4, ensure_ascii=False)

    # 4. 保存结果
    with open(args.descriptions_json, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=4, ensure_ascii=False)
    print(f"{args.descriptions_json} 已成功更新，每帧下添加了 'description' 和 'questions' 键。")

    backup_path = args.descriptions_json.replace(".json", "_questions.json")
    with open(backup_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=4, ensure_ascii=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量生成视频帧caption和子问题并更新 CRAD JSON (InternVL3-8B)")
    parser.add_argument("--data_root", type=str, 
                        default="/root/Ours_in_TAD/module_2/result_sample_TAD_Q15", help="图片根目录，folder和image拼接")
    parser.add_argument("--descriptions_json", type=str, 
                        default="/root/Ours_in_TAD/module_3/result_VAD_TAD_012_Q15/result_TAD_Q15.json", 
                        help="包含视频帧描述的 JSON 文件路径（结构为folder->image->info）")
    parser.add_argument("--guiding_questions_json", type=str, 
                        default="/root/Ours_in_TAD/module_3/guiding_questions_TADv4_15.json", 
                        help="包含指导问题的 JSON 文件路径")
    parser.add_argument("--model_id", type=str, default="/root/autodl-tmp/InternVL3-8B", help="大模型路径")
    args = parser.parse_args()
    main(args)

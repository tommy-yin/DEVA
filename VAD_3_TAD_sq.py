import cv2
import os
import glob
import json
import random
from tqdm import tqdm
from PIL import Image
import math
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import re
import warnings
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.ndimage import gaussian_filter1d
import datetime

warnings.filterwarnings("ignore", category=UserWarning)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

####################################
# 模型、图片预处理等函数
####################################


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


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_ar)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0]*x[1])
    target_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = int(image_size * target_ratio[0])
    target_height = int(image_size * target_ratio[1])
    blocks = target_ratio[0] * target_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    grid_w = target_width // image_size
    for i in range(blocks):
        box = (
            (i % grid_w) * image_size,
            (i // grid_w) * image_size,
            ((i % grid_w) + 1) * image_size,
            ((i // grid_w) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, max_num=max_num, use_thumbnail=True)  # 是否用 thumbnail (False效果好)
    # 对每一个 tile 进行 transform，并堆叠成一个张量
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values  # 返回形状为 (tiles, C, H, W)


####################################
# 工具函数
####################################

def print_gpu_memory_usage(step_name=""):
    """
    打印当前GPU显存使用情况
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3     # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"{step_name} - GPU Memory: {allocated:.2f}GB/{total:.2f}GB allocated, {cached:.2f}GB cached")
    else:
        print(f"{step_name} - No CUDA available")


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


def parse_result(answer):
    """
    从回复中提取异常分数，范围为 0.1 到 0.9。
    优先根据正则匹配，匹配格式例如: "Output: [0.3]" 或 "Output: 0.7" 或其他包含 0.1-0.9 的格式
    如果未匹配，则返回默认异常分数 0.1。
    """
    import re
    # 首先尝试匹配 Output: 格式
    match = re.search(r'Output:\s*\[?\s*(0\.[1-9])\s*\]?', answer)
    if match:
        return float(match.group(1))
    
    # 尝试匹配任何 0.1 到 0.9 的数字
    match = re.search(r'\b(0\.[1-9])\b', answer)
    if match:
        return float(match.group(1))
    
    # 尝试匹配描述性的分数
    lower_answer = answer.lower()
    if any(word in lower_answer for word in ['low', 'normal', 'minimal', 'unlikely']):
        return 0.1
    elif any(word in lower_answer for word in ['moderate', 'medium', 'possible']):
        return 0.5
    elif any(word in lower_answer for word in ['high', 'likely', 'suspicious', 'anomal']):
        return 0.8
    
    # 如果都没有匹配到，返回默认值
    return 0.1





####################################
# 关键函数
####################################


def vqa_multi_round_inference(image_path, questions_list, model, processor, max_new_tokens=512):
    """
    单图多轮对话的VQA函数
    
    参数：
        image_path: 图片路径
        questions_list: 问题列表，每个问题为字符串
        model: 模型实例
        processor: 处理器实例（tokenizer）
        max_new_tokens: 最大生成token数
    
    返回：
        responses: 每轮对话的响应列表
        history: 最终的对话历史
    """
    # 预处理图片，得到 pixel_values 张量，并送入到模型
    pixel_values = load_image(image_path).to(torch.bfloat16).to(model.device)
    
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9, top_k=50)
    
    responses = []
    history = None
    
    try:
        for i, question in enumerate(questions_list):
            # 第一轮对话需要包含图片标识符
            if i == 0:
                formatted_question = f'<image>\n{question}'
            else:
                formatted_question = question
            formatted_question = f'{formatted_question} Please answer with 2-3 sentences.'
            
            # 进行对话
            response, history = model.chat(
                processor, 
                pixel_values, 
                formatted_question, 
                generation_config, 
                history=history, 
                return_history=True
            )
            
            responses.append(response)
            
            # 每轮对话后清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    finally:
        # 确保清理显存
        del pixel_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return responses, history


def process_video_frames(root_folder, model, processor, guiding_questions_dict, result_file_path):
    """
    Processes video frames from root_folder.
    
    For each frame, it extracts the video folder from the frame path,
    then uses questions from guiding_questions_dict for multi-round conversation.
    
    Now saves results after each image processing to prevent data loss.
    
    Parameters:
        result_file_path: Path to save/load the JSON results file
    
    Returns:
      results: dict mapping video_folder to frame scores and responses.
    """
    # 尝试加载已有结果
    results = {}
    if os.path.exists(result_file_path):
        try:
            with open(result_file_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"✅ 加载已有结果文件: {result_file_path}")
            print(f"📊 已有 {len(results)} 个视频文件夹的结果")
        except Exception as e:
            print(f"⚠️  加载已有结果失败: {e}，从头开始处理")
            results = {}
    else:
        print(f"📝 创建新的结果文件: {result_file_path}")
        # 确保目录存在
        os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
    
    # 获取指定数量的帧
    frames = get_image_paths(root_folder)
    print(f"获取到 {len(frames)} 个帧进行处理")
    
    # 统计已处理的帧数
    processed_count = 0
    for video_folder, frames_data in results.items():
        for frame_key, frame_data in frames_data.items():
            if isinstance(frame_data, dict) and 'reason' in frame_data and 'anomaly_score' in frame_data:
                processed_count += 1
    
    print(f"📈 已处理 {processed_count} 个帧，剩余 {len(frames) - processed_count} 个帧")
    
    # 追加的异常分数问题
    anomaly_score_question = (
        "Using your reasoning above, pick an anomaly score for the frame from {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}, "
        "If any anomalies or violations of security rules are detected, a relatively high score is assigned to reflect potential risks.\n"
        "where lower means normal and higher means abnormal. Format strictly as:\n"
        "• Answers: [your brief reasoning]\n"
        "• Output Anomaly Score: [one value from the list above].\n"
        " No extra comments."
    )

    frames_processed_this_run = 0
    
    for idx, img_path in enumerate(tqdm(frames, desc="Processing frames")):
        video_folder = os.path.dirname(img_path)
        frame_key = os.path.basename(img_path)
        
        # 检查是否已经处理过这个帧
        if (video_folder in results and 
            frame_key in results[video_folder] and 
            isinstance(results[video_folder][frame_key], dict) and
            'reason' in results[video_folder][frame_key] and 
            'anomaly_score' in results[video_folder][frame_key]):
            print(f"⏭️  跳过已处理帧: {video_folder}/{frame_key}")
            continue
        
        # 显示当前显存使用情况
        if torch.cuda.is_available() and frames_processed_this_run % 5 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3     # GB
            print(f"Frame {idx+1}/{len(frames)} - GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
        
        # Ensure this video folder has an entry.
        if video_folder not in results:
            results[video_folder] = {}
        
        # 获取该视频文件夹对应的问题
        questions_nested = guiding_questions_dict.get(video_folder, {}).get(frame_key, {}).get("questions", [])
        print(questions_nested)

        # 将嵌套的问题列表扁平化
        questions_list = []
        for group in questions_nested:
            if isinstance(group, list):
                questions_list.extend(group)
            else:
                questions_list.append(group)
        
        # 如果没有问题，使用默认问题
        if not questions_list:
            print(f"Warning: No questions found for {video_folder}/{frame_key}, using default questions.")
            questions_list = ["Please describe the image in detail."]
        
        # 确保问题列表至少有一个问题
        # questions_list.append("Please describe the image in detail.")

        # 追加异常分数问题
        questions_list.append(anomaly_score_question)
        
        try:
            print(f"🔄 处理帧: {video_folder}/{frame_key} ({idx+1}/{len(frames)})")
            
            # 使用多轮对话函数
            responses, history = vqa_multi_round_inference(
                image_path=img_path,
                questions_list=questions_list,
                model=model,
                processor=processor,
                max_new_tokens=512
            )
            
            # 保存完整的对话历史
            if frame_key not in results[video_folder]:
                results[video_folder][frame_key] = {}
            results[video_folder][frame_key]['reason'] = history
            
            # 从最后一个回答中解析异常分数
            last_response = responses[-1] if responses else ""
            anomaly_score = parse_result(last_response)
            results[video_folder][frame_key]['anomaly_score'] = anomaly_score
            
            print(f"✅ 完成帧处理，异常分数: {anomaly_score}")
            
            # 立即保存结果到文件
            try:
                with open(result_file_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                print(f"💾 结果已保存到: {result_file_path}")
            except Exception as save_error:
                print(f"⚠️  保存结果失败: {save_error}")
            
            # 清理临时变量
            del responses, history, last_response
            frames_processed_this_run += 1
            
        except Exception as e:
            print(f"❌ 处理帧 {img_path} 时出错: {e}")
            # 发生错误时设置默认值
            if frame_key not in results[video_folder]:
                results[video_folder][frame_key] = {}
            results[video_folder][frame_key]['reason'] = f"Error processing frame: {str(e)}"
            results[video_folder][frame_key]['anomaly_score'] = 0.1
            
            # 即使出错也保存结果
            try:
                with open(result_file_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                print(f"💾 错误处理结果已保存")
            except Exception as save_error:
                print(f"⚠️  保存错误结果失败: {save_error}")
        
        finally:
            # 每处理完一张图片后强制清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 每处理5张图片后进行垃圾回收
            if frames_processed_this_run % 5 == 0:
                import gc
                gc.collect()

    print(f"🎉 总共处理了 {len(frames)} 个帧，来自 {len(results)} 个视频文件夹")
    print(f"📈 本次运行新处理了 {frames_processed_this_run} 个帧")
    return results


def main():
    args = parse_args()
    
    # 清理显存缓存
    torch.cuda.empty_cache()
    print_gpu_memory_usage("Initial")
    
    model, processor = setup_model(args.model_id)
    print_gpu_memory_usage("After loading model")
    
    # ---------- Load CRAD_results.json for guiding questions ----------
    # 检查是否已有结果文件，如果有则直接使用，否则需要先运行VAD_1和VAD_2
    if not os.path.exists(args.predict_result_dir):
        print(f"⚠️  结果文件不存在: {args.predict_result_dir}")
        print("请确保已运行VAD_1_frame_caption.py和VAD_2_route_plan.py生成初始结果文件")
        return
    
    with open(args.predict_result_dir, "r", encoding="utf-8") as f:
        crad_results = json.load(f)
    
    print(f"✅ 加载CRAD结果文件: {args.predict_result_dir}")
    print(f"📊 包含 {len(crad_results)} 个视频文件夹")
    
    # Pass the complete CRAD results to have access to questions for each video
    guiding_questions_dict = crad_results
    
    # ---------- Model Inference ----------
    # 这个函数现在会自动保存每个处理完的图片结果
    final_results = process_video_frames(args.data_root, model, processor, guiding_questions_dict, args.predict_result_dir)
    
    print_gpu_memory_usage("After processing all frames")
    
    print(f"🎉 所有帧处理完成！结果已实时保存到: {args.predict_result_dir}")
    
    # 最终清理
    torch.cuda.empty_cache()
    print_gpu_memory_usage("Final cleanup")


def parse_args():
    parser = argparse.ArgumentParser(description="异常视频帧检测与分类 - InternVL3-8B")
    parser.add_argument("--data_root", type=str, default="/root/Ours_in_TAD/module_2/result_sample_TAD_Q15", help="采样帧列表文件路径")
    parser.add_argument("--predict_result_dir", type=str, 
                        default="/root/Ours_in_TAD/module_3/result_VAD_TAD_012_Q15/result_TAD_Q15.json", 
                        help="保存结果文件路径")
    parser.add_argument("--model_id", type=str, default="/root/autodl-tmp/InternVL3-8B", help="使用的模型ID或路径")
    parser.add_argument('--is_validation', type=lambda x: x.lower() == "true", default=False, help='是否 ground truth 校验')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.ndimage import gaussian_filter1d
import argparse
import re
import csv


def parse_ground_truth(gt_file):
    """
    解析 ground truth 文件，返回字典，键为视频文件夹名称，值为包含总帧数和异常区间的字典
    文件格式：video_name total_frames start1 end1 [start2 end2 ...]
    例如：
      01_Accident_004.mp4 190 53 118
      Normal_012.mp4 70 -1 -1
    """
    gt_ranges = {}
    if not os.path.exists(gt_file):
        print(f"Ground truth 文件 {gt_file} 不存在！")
        return gt_ranges
    
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 4:
                continue
            
            video_name = parts[0]  # e.g., "01_Accident_004.mp4"
            total_frames = int(parts[1])
            
            # 去掉.mp4后缀作为键
            video_key = os.path.splitext(video_name)[0]  # e.g., "01_Accident_004"
            
            anomaly_regions = []
            # 解析异常区间，成对读取
            for i in range(2, len(parts), 2):
                if i + 1 < len(parts):
                    start = int(parts[i])
                    end = int(parts[i + 1])
                    # -1 -1 表示正常视频
                    if start != -1 and end != -1:
                        anomaly_regions.append((start, end))
            
            gt_ranges[video_key] = {
                'total_frames': total_frames,
                'anomaly_regions': anomaly_regions
            }
    
    return gt_ranges


def extract_frame_number(frame_name):
    """
    从帧文件名中提取帧号
    例如: "frame_0080.jpg" -> 80
    """
    match = re.search(r'(\d+)', frame_name)
    if match:
        return int(match.group(1))
    return 0


def load_crad_results(crad_file):
    """
    从CRAD结果文件中加载异常分数
    返回格式: {video_folder: {frame_number: anomaly_score}}
    """
    with open(crad_file, 'r', encoding='utf-8') as f:
        crad_data = json.load(f)
    
    results = {}
    for video_folder, data in crad_data.items():
        if isinstance(data, dict) and any(key.endswith('.jpg') or key.endswith('.png') for key in data.keys()):
            results[video_folder] = {}
            for frame_name, frame_data in data.items():
                if frame_name.endswith(('.jpg', '.png')) and isinstance(frame_data, dict):
                    if 'anomaly_score' in frame_data:
                        frame_number = extract_frame_number(frame_name)
                        results[video_folder][frame_number] = frame_data['anomaly_score']
    
    return results


def load_weight_scores(video_name, scores_dir="/root/my_VAD/demo_module2_sampling/result_sample_5"):
    """
    加载指定视频的contextual和semantic分数
    
    参数:
        video_name: 视频名称（如 "031"）
        scores_dir: scores文件目录
    
    返回:
        frame_names, context_scores, semantic_scores
    """
    scores_file = os.path.join(scores_dir, f"{video_name}.mp4_scores.json")
    
    if not os.path.exists(scores_file):
        print(f"未找到权重分数文件: {scores_file}")
        return None, None, None
    
    try:
        with open(scores_file, 'r', encoding='utf-8') as f:
            scores_data = json.load(f)
        
        frame_names = scores_data.get('frame_names', [])
        context_scores = scores_data.get('context_scores_norm', [])
        semantic_scores = scores_data.get('semantic_scores_norm', [])
        
        # print(f"成功加载权重分数文件: {scores_file}")
        # print(f"  - 帧数: {len(frame_names)}")
        # print(f"  - Contextual分数: {len(context_scores)}")
        # print(f"  - Semantic分数: {len(semantic_scores)}")
        
        return frame_names, context_scores, semantic_scores
        
    except Exception as e:
        print(f"加载权重分数文件失败: {e}")
        return None, None, None


def interpolate_scores_to_full_video(frame_scores, total_frames, normalize=True):
    """
    将采样帧的异常分数线性插值到完整视频的所有帧
    
    参数:
        frame_scores: {frame_number: anomaly_score}
        total_frames: 视频总帧数（从ground truth获取）
        normalize: 是否进行归一化，默认为True
    
    返回:
        full_indices: 完整帧索引
        interpolated_scores: 插值后的异常分数
    """
    if not frame_scores:
        print(f"没有找到异常分数数据")
        return None, None
    
    # 提取并排序采样帧
    sample_indices = sorted(frame_scores.keys())
    sample_scores = [frame_scores[idx] for idx in sample_indices]
    
    # 创建完整帧索引
    full_indices = np.arange(0, total_frames)
    
    # 线性插值
    interpolated_scores = np.interp(full_indices, sample_indices, sample_scores)
    
    # 根据参数决定是否归一化到0-1区间
    if normalize:
        min_score = np.min(interpolated_scores)
        max_score = np.max(interpolated_scores)
        if max_score > min_score:
            interpolated_scores = (interpolated_scores - min_score) / (max_score - min_score)
    
    # 高斯平滑
    smoothed_scores = gaussian_filter1d(interpolated_scores, sigma=5)
    
    return full_indices, smoothed_scores


def interpolate_scores_to_full_video_weighted(frame_scores, total_frames, video_name, normalize=True, scores_dir="/root/my_VAD/demo_module2_sampling/result_sample_5", distance_weight_factor=2.0, context_weight_factor=1.0, semantic_weight_factor=1.5):
    """
    使用contextual和semantic分数作为权重，将采样帧的异常分数补全到完整视频的所有帧
    
    参数:
        frame_scores: {frame_number: anomaly_score} 采样帧的异常分数
        total_frames: 视频总帧数（从ground truth获取）
        video_name: 视频名称（用于加载权重分数）
        normalize: 是否进行归一化，默认为True
        scores_dir: 权重分数文件目录
        distance_weight_factor: 距离权重因子，默认为2.0
        context_weight_factor: contextual权重因子，默认为1.0
        semantic_weight_factor: semantic权重因子，默认为1.5
    
    返回:
        full_indices: 完整帧索引
        interpolated_scores: 补全后的异常分数
    """
    if not frame_scores:
        print(f"没有找到异常分数数据")
        return None, None
    
    # 加载权重分数
    frame_names, context_scores, semantic_scores = load_weight_scores(video_name, scores_dir)
    
    if frame_names is None or context_scores is None or semantic_scores is None:
        print(f"权重分数加载失败，回退到线性插值方法")
        return interpolate_scores_to_full_video(frame_scores, total_frames, normalize)
    
    # 确保权重分数的长度与视频帧数匹配
    if len(context_scores) == total_frames - 1 and len(semantic_scores) == total_frames - 1:
        # 如果权重分数比视频帧数少1（通常情况），在末尾添加最后一个值
        context_scores.append(context_scores[-1])
        semantic_scores.append(semantic_scores[-1])
        print(f"权重分数长度调整：扩展到 {len(context_scores)} 帧")
    elif len(context_scores) != total_frames or len(semantic_scores) != total_frames:
        print(f"权重分数长度不匹配：context={len(context_scores)}, semantic={len(semantic_scores)}, video_frames={total_frames}")
        print(f"回退到线性插值方法")
        return interpolate_scores_to_full_video(frame_scores, total_frames, normalize)
    
    # 提取并排序采样帧
    sample_indices = sorted(frame_scores.keys())
    sample_scores = [frame_scores[idx] for idx in sample_indices]
    
    # print(f"开始基于权重的分数补全...")
    # print(f"  - 采样帧数: {len(sample_indices)}")
    # print(f"  - 总帧数: {total_frames}")
    # print(f"  - 采样帧索引: {sample_indices}")
    
    # 创建完整帧索引
    full_indices = np.arange(0, total_frames)
    interpolated_scores = np.zeros(total_frames)
    
    # 转换为numpy数组以便计算
    context_scores = np.array(context_scores)
    semantic_scores = np.array(semantic_scores)
    sample_indices = np.array(sample_indices)
    sample_scores = np.array(sample_scores)
    
    # 对每一帧计算异常分数
    for frame_idx in range(total_frames):
        if frame_idx in frame_scores:
            # 如果是采样帧，直接使用原始分数
            interpolated_scores[frame_idx] = frame_scores[frame_idx]
        else:
            # 对于非采样帧，使用权重方法计算分数
            weights = np.zeros(len(sample_indices))
            
            for i, sample_idx in enumerate(sample_indices):
                # 1. 距离权重：距离越近权重越大
                distance = abs(frame_idx - sample_idx) + 1  # 避免除零
                distance_weight = 1.0 / (distance ** distance_weight_factor)
                
                # 2. Contextual权重：考虑当前帧与采样帧之间的contextual相似性
                # 取当前帧和采样帧之间路径上的contextual分数的平均值
                start_idx = min(frame_idx, sample_idx)
                end_idx = max(frame_idx, sample_idx)
                if start_idx == end_idx:
                    context_weight = context_scores[frame_idx]
                else:
                    context_weight = np.mean(context_scores[start_idx:end_idx+1])
                
                # 3. Semantic权重：当前帧的semantic分数
                semantic_weight = semantic_scores[frame_idx]
                
                # 4. 采样帧的semantic分数也考虑进来
                sample_semantic_weight = semantic_scores[sample_idx]
                
                # 综合权重计算
                # 距离权重 × contextual相似度 × 当前帧semantic权重 × 采样帧semantic权重
                combined_weight = (distance_weight * 
                                 (1 + context_weight * context_weight_factor) * 
                                 (1 + semantic_weight * semantic_weight_factor * sample_semantic_weight))
                
                weights[i] = combined_weight
            
            # 归一化权重
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                # 加权平均计算该帧的异常分数
                interpolated_scores[frame_idx] = np.sum(weights * sample_scores)
            else:
                # 如果权重全为0，使用最近邻方法
                nearest_idx = np.argmin(np.abs(sample_indices - frame_idx))
                interpolated_scores[frame_idx] = sample_scores[nearest_idx]
    
    print(f"权重补全完成")
    # print(f"  - 补全后分数范围: [{np.min(interpolated_scores):.4f}, {np.max(interpolated_scores):.4f}]")
    
    # 根据参数决定是否归一化到0-1区间
    if normalize:
        min_score = np.min(interpolated_scores)
        max_score = np.max(interpolated_scores)
        if max_score > min_score:
            interpolated_scores = (interpolated_scores - min_score) / (max_score - min_score)
            # print(f"  - 归一化后分数范围: [{np.min(interpolated_scores):.4f}, {np.max(interpolated_scores):.4f}]")
    
    # 高斯平滑
    smoothed_scores = gaussian_filter1d(interpolated_scores, sigma=5)
    print(f"  - 高斯平滑完成，sigma=5")
    
    return full_indices, smoothed_scores


def plot_anomaly_curve(full_indices, smoothed_scores, video_name, gt_ranges, output_dir, original_frame_scores=None, video_metrics=None):
    """
    绘制异常分数曲线图，根据 ground truth 区间用红色透明背景标示
    
    参数:
        original_frame_scores: 原始采样帧的异常分数 {frame_number: anomaly_score}
        video_metrics: 视频指标字典，包含AUROC和AUPR等信息
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制插值后的平滑曲线
    plt.plot(full_indices, smoothed_scores, 'b-', linewidth=1.5, label='Interpolated Anomaly Score', alpha=0.8)
    
    # 绘制原始采样点
    if original_frame_scores:
        sample_indices = list(original_frame_scores.keys())
        sample_scores = list(original_frame_scores.values())
        plt.scatter(sample_indices, sample_scores, color='red', s=30, zorder=5, 
                   label='Original Sample Points', alpha=0.9)
    
    plt.xlabel('Frame Index')
    plt.ylabel('Anomaly Score')
    
    # 构建包含指标的标题
    if video_metrics and video_metrics['status'] == 'success':
        title = f'Anomaly Detection Results for Video {video_name}\nAUROC: {video_metrics["auroc"]:.4f} | AUPR: {video_metrics["aupr"]:.4f} | Anomaly Frames: {video_metrics["anomaly_frames"]}/{video_metrics["total_frames"]} ({video_metrics["anomaly_ratio"]*100:.1f}%)'
    else:
        # 对于无法计算指标的视频，使用简单的标题
        title = f'Anomaly Detection Results for Video {video_name}'
    
    plt.title(title, fontsize=11)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # 标注异常区间
    if video_name in gt_ranges and gt_ranges[video_name]['anomaly_regions']:
        for i, (start, end) in enumerate(gt_ranges[video_name]['anomaly_regions']):
            plt.axvspan(start, end, color='red', alpha=0.2, 
                       label='Ground Truth Anomaly' if i == 0 else "")
    
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"anomaly_curve_{video_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"异常曲线已保存: {save_path}")


def compute_metrics(crad_results, gt_file, output_dir, normalize=True, use_weighted_interpolation=True, scores_dir="/root/my_VAD/demo_module2_sampling/result_sample_5", distance_weight_factor=2.0, context_weight_factor=1.0, semantic_weight_factor=1.5):
    """
    计算异常检测指标并生成可视化图表
    
    参数:
        crad_results: CRAD结果数据
        gt_file: ground truth文件路径
        output_dir: 输出目录
        normalize: 是否进行归一化，默认为True
        use_weighted_interpolation: 是否使用权重补全方法，默认为True
        scores_dir: 权重分数文件目录
        distance_weight_factor: 距离权重因子，默认为2.0
        context_weight_factor: contextual权重因子，默认为1.0
        semantic_weight_factor: semantic权重因子，默认为1.5
    
    返回:
        auroc, aupr: 评估指标
    """
    gt_dict = parse_ground_truth(gt_file)
    
    # 添加调试信息
    print(f"Ground truth 包含 {len(gt_dict)} 个视频:")
    gt_keys = list(gt_dict.keys())[:5]  # 显示前5个
    print(f"Ground truth 示例键: {gt_keys}")
    
    crad_keys = list(crad_results.keys())[:5]  # 显示前5个
    crad_names = [os.path.basename(key) for key in crad_keys]
    crad_names_no_ext = [os.path.splitext(name)[0] for name in crad_names]
    print(f"CRAD results 示例键: {crad_keys}")
    print(f"CRAD basename: {crad_names}")
    print(f"CRAD basename (no ext): {crad_names_no_ext}")
    print()
    
    y_true_all = []
    y_scores_all = []
    
    # 添加只包含异常视频的数据收集
    y_true_anomaly_only = []
    y_scores_anomaly_only = []
    
    total_videos = 0
    processed_videos = 0
    total_sample_frames = 0
    
    # 统计异常视频数量
    anomaly_videos_processed = 0
    
    # 存储每个视频的详细指标
    video_metrics = {}
    
    print(f"使用补全方法: {'权重补全' if use_weighted_interpolation else '线性插值'}")
    
    for video_folder, frame_scores in crad_results.items():
        video_name_full = os.path.basename(video_folder)  # 可能包含.mp4后缀
        video_name = os.path.splitext(video_name_full)[0]  # 去掉.mp4后缀，与gt_dict的键格式一致
        total_videos += 1
        
        if video_name not in gt_dict:
            print(f"警告: 未找到视频 {video_name_full} 的ground truth信息")
            continue
        
        # 从ground truth获取总帧数
        total_frames = gt_dict[video_name]['total_frames']
        
        # 选择补全方法
        if use_weighted_interpolation:
            full_indices, smoothed_scores = interpolate_scores_to_full_video_weighted(
                frame_scores, total_frames, video_name, normalize, scores_dir, distance_weight_factor, context_weight_factor, semantic_weight_factor)
        else:
            full_indices, smoothed_scores = interpolate_scores_to_full_video(
                frame_scores, total_frames, normalize)
        
        if full_indices is None or smoothed_scores is None:
            print(f"跳过视频 {video_name_full}（补全失败）")
            continue
        
        # 计算标签
        segments = gt_dict[video_name]['anomaly_regions']
        y_true = np.zeros(len(full_indices))
        for start, end in segments:
            y_true[start:end+1] = 1
        
        # 计算单个视频的指标
        video_auroc = 0.0
        video_aupr = 0.0
        video_status = "success"
        video_error_msg = ""
        
        try:
            # 检查是否有足够的数据计算指标
            unique_labels = np.unique(y_true)
            if len(unique_labels) > 1:  # 既有正常帧又有异常帧
                video_auroc = roc_auc_score(y_true, smoothed_scores)
                precision, recall, _ = precision_recall_curve(y_true, smoothed_scores)
                video_aupr = auc(recall, precision)
                print(f"✅ 视频 {video_name_full}: AUROC={video_auroc:.4f}, AUPR={video_aupr:.4f}")
            else:
                # 全部是正常帧或全部是异常帧
                video_status = "skipped"
                if unique_labels[0] == 0:
                    video_error_msg = "全部为正常帧，无法计算指标"
                else:
                    video_error_msg = "全部为异常帧，无法计算指标"
                print(f"⚠️  视频 {video_name_full}: {video_error_msg}")
        
        except Exception as e:
            video_status = "error"
            video_error_msg = str(e)
            print(f"❌ 计算视频 {video_name_full} 指标时出错: {e}")
        
        # 构建当前视频的指标字典
        current_video_metrics = {
            'auroc': video_auroc,
            'aupr': video_aupr,
            'status': video_status,
            'error_message': video_error_msg,
            'total_frames': len(full_indices),
            'sample_frames': len(frame_scores),
            'anomaly_frames': int(np.sum(y_true)),
            'anomaly_ratio': float(np.sum(y_true) / len(y_true)),
            'anomaly_segments': len(segments),
            'score_range': {
                'min': float(np.min(smoothed_scores)),
                'max': float(np.max(smoothed_scores)),
                'mean': float(np.mean(smoothed_scores)),
                'std': float(np.std(smoothed_scores))
            }
        }
        
        # 绘制异常曲线，传递计算好的指标
        plot_anomaly_curve(full_indices, smoothed_scores, video_name, gt_dict, output_dir, 
                          original_frame_scores=frame_scores, video_metrics=current_video_metrics)
        
        # 保存视频级别的详细信息（使用video_name作为键，与gt_dict保持一致）
        video_metrics[video_name] = current_video_metrics
        
        # 收集数据用于整体评估
        y_true_all.extend(y_true)
        y_scores_all.extend(smoothed_scores)
        
        # 添加只包含异常视频的数据收集（视频名称中没有"Normal"的为异常视频）
        if "Normal" not in video_name:
            y_true_anomaly_only.extend(y_true)
            y_scores_anomaly_only.extend(smoothed_scores)
            anomaly_videos_processed += 1
        
        processed_videos += 1
        total_sample_frames += len(frame_scores)
        
        # print(f"处理视频 {video_name_full}: {len(frame_scores)} 个采样帧 -> {len(full_indices)} 个完整帧")
    
    # 计算视频级别统计
    successful_videos = [v for v in video_metrics.values() if v['status'] == 'success']
    if successful_videos:
        video_aurocs = [v['auroc'] for v in successful_videos]
        video_auprs = [v['aupr'] for v in successful_videos]
        
        video_level_stats = {
            'mean_auroc': float(np.mean(video_aurocs)),
            'std_auroc': float(np.std(video_aurocs)),
            'mean_aupr': float(np.mean(video_auprs)),
            'std_aupr': float(np.std(video_auprs)),
            'successful_videos': len(successful_videos),
            'skipped_videos': len([v for v in video_metrics.values() if v['status'] == 'skipped']),
            'error_videos': len([v for v in video_metrics.values() if v['status'] == 'error'])
        }
    else:
        video_level_stats = {
            'mean_auroc': 0.0,
            'std_auroc': 0.0,
            'mean_aupr': 0.0,
            'std_aupr': 0.0,
            'successful_videos': 0,
            'skipped_videos': len([v for v in video_metrics.values() if v['status'] == 'skipped']),
            'error_videos': len([v for v in video_metrics.values() if v['status'] == 'error'])
        }
    
    # 计算整体指标
    if len(y_true_all) > 0 and len(set(y_true_all)) > 1:
        print(f"y_scores_all: {len(y_scores_all)}")
        try:
            overall_auroc = roc_auc_score(y_true_all, y_scores_all)
            precision, recall, _ = precision_recall_curve(y_true_all, y_scores_all)
            overall_aupr = auc(recall, precision)
            
            # 计算只包含异常视频的指标
            anomaly_only_auroc = 0.0
            anomaly_only_aupr = 0.0
            if len(y_true_anomaly_only) > 0 and len(set(y_true_anomaly_only)) > 1:
                anomaly_only_auroc = roc_auc_score(y_true_anomaly_only, y_scores_anomaly_only)
                precision_anomaly, recall_anomaly, _ = precision_recall_curve(y_true_anomaly_only, y_scores_anomaly_only)
                anomaly_only_aupr = auc(recall_anomaly, precision_anomaly)
            
            print(f"\n" + "="*80)
            print(f"详细评估结果:")
            print(f"="*80)
            
            # 显示视频级别统计
            print(f"📊 视频级别统计:")
            print(f"   成功计算指标的视频: {video_level_stats['successful_videos']}")
            print(f"   跳过的视频（单一标签）: {video_level_stats['skipped_videos']}")
            print(f"   错误的视频: {video_level_stats['error_videos']}")
            if video_level_stats['successful_videos'] > 0:
                print(f"   视频平均AUROC: {video_level_stats['mean_auroc']:.4f} ± {video_level_stats['std_auroc']:.4f}")
                print(f"   视频平均AUPR: {video_level_stats['mean_aupr']:.4f} ± {video_level_stats['std_aupr']:.4f}")
            
            print(f"\n📊 整体统计:")
            print(f"   处理视频数: {processed_videos}/{total_videos}")
            print(f"   其中异常视频数: {anomaly_videos_processed}")
            print(f"   正常视频数: {processed_videos - anomaly_videos_processed}")
            print(f"   总采样帧数: {total_sample_frames}")
            print(f"   总完整帧数: {len(y_true_all)}")
            print(f"   异常帧数: {int(sum(y_true_all))}")
            print(f"   异常帧比例: {sum(y_true_all)/len(y_true_all)*100:.2f}%")
            print(f"   归一化设置: {'开启' if normalize else '关闭'}")
            
            print(f"\n🔍 全部数据整体指标:")
            print(f"   整体AUROC (全部视频): {overall_auroc:.4f}")
            print(f"   整体AUPR (全部视频): {overall_aupr:.4f}")
            
            if len(y_true_anomaly_only) > 0:
                print(f"\n🎯 异常视频整体指标:")
                print(f"   异常视频完整帧数: {len(y_true_anomaly_only)}")
                print(f"   异常视频异常帧数: {int(sum(y_true_anomaly_only))}")
                print(f"   异常视频异常帧比例: {sum(y_true_anomaly_only)/len(y_true_anomaly_only)*100:.2f}%")
                if len(set(y_true_anomaly_only)) > 1:
                    print(f"   整体AUROC (仅异常视频): {anomaly_only_auroc:.4f}")
                    print(f"   整体AUPR (仅异常视频): {anomaly_only_aupr:.4f}")
                else:
                    print(f"   异常视频数据标签单一，无法计算指标")
            else:
                print(f"\n🎯 异常视频整体指标:")
                print(f"   没有处理到异常视频数据")
            
            print(f"\n📋 各视频详细指标:")
            print(f"{'-'*80}")
            print(f"{'视频名称':<20} {'AUROC':<8} {'AUPR':<8} {'状态':<10} {'异常帧':<8} {'总帧数':<8}")
            print(f"{'-'*80}")
            for video_name, metrics in video_metrics.items():
                status_display = {
                    'success': '✅成功',
                    'skipped': '⚠️跳过',
                    'error': '❌错误'
                }.get(metrics['status'], metrics['status'])
                
                if metrics['status'] == 'success':
                    print(f"{video_name:<20} {metrics['auroc']:<8.4f} {metrics['aupr']:<8.4f} "
                          f"{status_display:<10} {metrics['anomaly_frames']:<8} {metrics['total_frames']:<8}")
                else:
                    print(f"{video_name:<20} {'N/A':<8} {'N/A':<8} "
                          f"{status_display:<10} {metrics['anomaly_frames']:<8} {metrics['total_frames']:<8}")
            
            print(f"="*80)
            
            # 保存评估结果
            evaluation_results = {
                'overall_metrics': {
                    'auroc_all_videos': overall_auroc,
                    'aupr_all_videos': overall_aupr,
                    'auroc_anomaly_only': anomaly_only_auroc,
                    'aupr_anomaly_only': anomaly_only_aupr,
                    'processed_videos': processed_videos,
                    'anomaly_videos_processed': anomaly_videos_processed,
                    'normal_videos_processed': processed_videos - anomaly_videos_processed,
                    'total_videos': total_videos,
                    'total_sample_frames': total_sample_frames,
                    'total_frames': len(y_true_all),
                    'anomaly_frames': int(sum(y_true_all)),
                    'anomaly_ratio': float(sum(y_true_all)/len(y_true_all)),
                    'anomaly_only_frames': len(y_true_anomaly_only),
                    'anomaly_only_anomaly_frames': int(sum(y_true_anomaly_only)) if len(y_true_anomaly_only) > 0 else 0,
                    'normalize_enabled': normalize,
                    'interpolation_method': 'weighted' if use_weighted_interpolation else 'linear'
                },
                'video_level_stats': video_level_stats,
                'individual_video_metrics': video_metrics
            }
            
            results_path = os.path.join(output_dir, 'evaluation_results.json')
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
            
            print(f"📁 评估结果已保存: {results_path}")
            
            # 另外保存一个CSV格式的视频级别指标表格
            csv_path = os.path.join(output_dir, 'video_metrics.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['video_name', 'auroc', 'aupr', 'status', 'total_frames', 'sample_frames', 
                             'anomaly_frames', 'anomaly_ratio', 'anomaly_segments', 'score_min', 
                             'score_max', 'score_mean', 'score_std', 'error_message']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for video_name, metrics in video_metrics.items():
                    row = {
                        'video_name': video_name,
                        'auroc': metrics['auroc'] if metrics['status'] == 'success' else 'N/A',
                        'aupr': metrics['aupr'] if metrics['status'] == 'success' else 'N/A',
                        'status': metrics['status'],
                        'total_frames': metrics['total_frames'],
                        'sample_frames': metrics['sample_frames'],
                        'anomaly_frames': metrics['anomaly_frames'],
                        'anomaly_ratio': f"{metrics['anomaly_ratio']:.4f}",
                        'anomaly_segments': metrics['anomaly_segments'],
                        'score_min': f"{metrics['score_range']['min']:.4f}",
                        'score_max': f"{metrics['score_range']['max']:.4f}",
                        'score_mean': f"{metrics['score_range']['mean']:.4f}",
                        'score_std': f"{metrics['score_range']['std']:.4f}",
                        'error_message': metrics['error_message']
                    }
                    writer.writerow(row)
            
            print(f"📊 视频指标CSV表格已保存: {csv_path}")
            
            return overall_auroc, overall_aupr, anomaly_only_auroc, anomaly_only_aupr
            
        except Exception as e:
            print(f"计算整体指标时出错: {e}")
            return 0, 0, 0, 0
    else:
        print("没有有效的数据进行评估")
        return 0, 0, 0, 0


def main():
    parser = argparse.ArgumentParser(description="异常检测结果评估和可视化")
    parser.add_argument("--crad_file", type=str, 
                       default="/root/Ours_in_TAD/module_3/result_VAD_TAD_017_Q10_v1/result_TAD_Q10_V1_4.json",
                       help="CRAD结果文件路径")
    parser.add_argument("--output_dir", type=str, 
                       default="/root/Ours_in_TAD/module_3/result_VAD_TAD_017_Q10_v1/evaluation_results_weighted",
                       help="输出目录")
    parser.add_argument("--gt_file", type=str, 
                       default="/root/Ours_in_TAD/TAD_dataset/vals.txt",
                       help="Ground truth文件路径")
    parser.add_argument("--normalize", action="store_true", default=False,
                       help="是否对异常分数进行归一化处理（默认：开启）")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false",
                       help="关闭异常分数归一化处理")
    parser.add_argument("--use_weighted_interpolation", action="store_true", default=True,
                       help="是否使用权重补全方法，默认为True")
    parser.add_argument("--no_weighted_interpolation", dest="use_weighted_interpolation", action="store_false",
                       help="关闭权重补全方法")
    parser.add_argument("--scores_dir", type=str,
                       default="/root/Ours_in_TAD/module_2/result_sample_TAD",
                       help="权重分数文件目录路径")
    parser.add_argument("--distance_weight_factor", type=float, default=1.0,
                       help="距离权重因子，默认为2.0")
    parser.add_argument("--context_weight_factor", type=float, default=2.0,
                       help="contextual权重因子，默认为1.0")
    parser.add_argument("--semantic_weight_factor", type=float, default=0.5,
                       help="semantic权重因子，默认为1.5")
    
    args = parser.parse_args()

    """
    使用示例:
    默认情况下归一化是开启的，权重补全是开启的：python VAD_4_metrics_visual_TAD_v2.py
    要关闭归一化：python VAD_4_metrics_visual_TAD_v2.py --no_normalize
    要开启归一化：python VAD_4_metrics_visual_TAD_v2.py --normalize
    要关闭权重补全（使用线性插值）：python VAD_4_metrics_visual_TAD_v2.py --no_weighted_interpolation
    同时设置多个参数：python VAD_4_metrics_visual_TAD_v2.py --normalize --no_weighted_interpolation
    
    权重参数调整示例:
    python VAD_4_metrics_visual_TAD_v2.py --distance_weight_factor 3.0 --semantic_weight_factor 2.0
    python VAD_4_metrics_visual_TAD_v2.py --scores_dir /path/to/scores --context_weight_factor 1.5
    """
    
    print("开始加载CRAD结果...")
    crad_results = load_crad_results(args.crad_file)
    print(f"加载了 {len(crad_results)} 个视频的结果")
    
    print(f"归一化设置: {'开启' if args.normalize else '关闭'}")
    print(f"补全方法: {'权重补全' if args.use_weighted_interpolation else '线性插值'}")
    if args.use_weighted_interpolation:
        print(f"权重参数设置:")
        print(f"  - 权重分数目录: {args.scores_dir}")
        print(f"  - 距离权重因子: {args.distance_weight_factor}")
        print(f"  - Contextual权重因子: {args.context_weight_factor}")
        print(f"  - Semantic权重因子: {args.semantic_weight_factor}")
    print("开始计算评估指标...")
    auroc, aupr, anomaly_only_auroc, anomaly_only_aupr = compute_metrics(crad_results, args.gt_file, args.output_dir, args.normalize, args.use_weighted_interpolation, args.scores_dir, args.distance_weight_factor, args.context_weight_factor, args.semantic_weight_factor)
    
    print(f"\n最终评估结果:")
    print(f"📊 全部视频指标:")
    print(f"   AUROC (All Videos): {auroc:.4f}")
    print(f"   AUPR (All Videos): {aupr:.4f}")
    print(f"🎯 异常视频指标:")
    print(f"   AUROC (Anomaly Videos Only): {anomaly_only_auroc:.4f}")
    print(f"   AUPR (Anomaly Videos Only): {anomaly_only_aupr:.4f}")


if __name__ == "__main__":
    main()

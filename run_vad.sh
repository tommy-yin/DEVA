#!/bin/bash


# 统一路径变量
# RESULT_DIR="/root/Ours_in_TAD/module_2/result_sample_TAD"
# DESCRIPTIONS_JSON="/root/Ours_in_TAD/module_3/result_VAD_TAD_017_Q10_v1/result_TAD_Q10_V1_5.json"
# GUIDING_QUESTIONS_JSON="/root/Ours_in_TAD/module_3/guiding_questions_TAD.json"
# EVAL_OUTPUT_DIR="/root/Ours_in_TAD/module_3/result_VAD_TAD_017_Q10_v1/evaluation_results_Q10_v5_weighted"


# echo "运行VAD1 2"
# python /root/Ours_in_TAD/module_3/VAD_1_2_TAD_sq.py \
#     --data_root "$RESULT_DIR" \
#     --descriptions_json "$DESCRIPTIONS_JSON" \
#     --guiding_questions_json "$GUIDING_QUESTIONS_JSON"

echo "运行VAD3"
python /root/Ours_in_TAD/module_3/VAD_3_TAD_sq.py \
    --data_root "/root/Ours_in_TAD/module_2/result_sample_TAD" \
    --predict_result_dir "/root/Ours_in_TAD/module_3/result_VAD_TAD_017_Q10_v1/result_TAD_Q10_V1_6.json"

echo "运行VAD4"
python /root/Ours_in_TAD/module_3/VAD_4_metrics_visual_TAD_v2.py \
    --crad_file "/root/Ours_in_TAD/module_3/result_VAD_TAD_017_Q10_v1/result_TAD_Q10_V1_6.json" \
    --output_dir "/root/Ours_in_TAD/module_3/result_VAD_TAD_017_Q10_v1/evaluation_results_Q10_v6_weighted" \
    --no_normalize

echo "运行VAD3"
python /root/Ours_in_TAD/module_3/VAD_3_TAD_sq.py \
    --data_root "/root/Ours_in_TAD/module_2/result_sample_TAD" \
    --predict_result_dir "/root/Ours_in_TAD/module_3/result_VAD_TAD_017_Q10_v1/result_TAD_Q10_V1_7.json"

echo "运行VAD4"
python /root/Ours_in_TAD/module_3/VAD_4_metrics_visual_TAD_v2.py \
    --crad_file "/root/Ours_in_TAD/module_3/result_VAD_TAD_017_Q10_v1/result_TAD_Q10_V1_7.json" \
    --output_dir "/root/Ours_in_TAD/module_3/result_VAD_TAD_017_Q10_v1/evaluation_results_Q10_v7_weighted" \
    --no_normalize



echo "完成所有运行！" 
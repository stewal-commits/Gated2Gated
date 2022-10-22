#!/bin/sh
export CUDA_VISIBLE_DEVICES=0

eval_files="./src/splits/gatedstereo/test_gatedstereo.txt"
python src/eval.py \
        --data_dir /external/10g/dense2/fs1/datasets/202210_GatedStereoDatasetv3 \
        --min_depth 0.1 \
        --max_depth 100.0 \
        --height  512 \
        --width   1024 \
        --load_weights_folder /external/10g/dense2/fs1/students/stewal/models/Gated2Gated/final_model_exp-19_weights_4 \
        --results_dir results/gatedstereo \
        --eval_files_path $eval_files \
        --dataset gatedstereo \
        --g2d_crop
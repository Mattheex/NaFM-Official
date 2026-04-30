#!/usr/bin/env bash

DATASET="External"
DATASET_ROOT="./downstream_data/External"
SAVE_INTERVAL=30

for SEED in 0 1 2; do
    for DATASET_ARG in 204 178 109 30 570 668 208; do
        for FOLD in 1 2 3 4; do
            FOLDER="./log-${DATASET}_${DATASET_ARG}_${FOLD}_${SEED}"

            if [ -d "${FOLDER}" ]; then
                echo "Removing existing directory: ${FOLDER}"
                rm -rf "${FOLDER}"
            fi

            echo "============================================"
            echo "Training NPT${DATASET_ARG}"
            echo "  log-dir   : ${FOLDER}"
            echo "  fold      : ${FOLD}"
            echo "  seed      : ${SEED}"
            echo "============================================"

            python train.py \
                --task          "finetune" \
                --log-dir       "${FOLDER}" \
                --dataset       "${DATASET}" \
                --dataset-arg   "${DATASET_ARG}" \
                --dataset-root  "${DATASET_ROOT}" \
                --save-interval "${SAVE_INTERVAL}" \
                --val-fold      "${FOLD}" \
                --seed          "${SEED}" \
                --lr             1.e-5 \
                --lr-min        1.e-6

            EXIT_CODE=$?
            if [ $EXIT_CODE -ne 0 ]; then
                echo "[WARNING] Training for ${DATASET_ARG} fold ${FOLD} exited with code ${EXIT_CODE}"
            fi
            echo ""
        done
    done
done

echo "All training finished."
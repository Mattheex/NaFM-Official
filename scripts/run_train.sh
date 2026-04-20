#!/usr/bin/env bash

DATASET="External"
DATASET_ROOT="./downstream_data/External"
SAVE_INTERVAL=10

for DATASET_ARG in 204 178 109 30 570 668 208; do
    for FOLD in 1 2 3 4; do
        FOLDER="./log-${DATASET}_${DATASET_ARG}_${FOLD}"

        echo "============================================"
        echo "Training NPT${DATASET_ARG}"
        echo "  log-dir   : ${FOLDER}"
        echo "  dataset   : ${DATASET}"
        echo "  fold      : ${FOLD}"
        echo "============================================"

        python train.py \
            --task          "finetune" \
            --log-dir       "${FOLDER}" \
            --dataset       "${DATASET}" \
            --dataset-arg   "${DATASET_ARG}" \
            --dataset-root  "${DATASET_ROOT}" \
            --save-interval "${SAVE_INTERVAL}" \
            --val-fold      "${FOLD}"

        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "[WARNING] Training for ${DATASET_ARG} fold ${FOLD} exited with code ${EXIT_CODE}"
        fi
        echo ""
    done
done

echo "All training finished."
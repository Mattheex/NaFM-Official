#!/usr/bin/env bash
# Run test.py for each sample-count folder under "ckpts/ckpt for matthieu/"
# Each folder contains: <N>samples.ckpt  splits.npz
#
# Usage:
#   bash scripts/run_tests_matthieu.sh
#
# Optional overrides (env variables):
#   DATASET         – dataset name   (default: Ontology)
#   DATASET_ARG     – dataset arg    (default: Class)
#   DATASET_ROOT    – dataset root   (default: ./downstream_data/Ontology)

DATASET="${DATASET:-Ontology}"
DATASET_ARG="${DATASET_ARG:-Class}"
DATASET_ROOT="${DATASET_ROOT:-./downstream_data/Ontology}"

BASE_DIR="ckpts/ckpt for matthieu"

for N in 4 8 16 24 40 64; do
    FOLDER="${BASE_DIR}/${N}samples"
    CKPT="${N}samples.ckpt"

    echo "============================================"
    echo "Testing ${N} samples"
    echo "  log-dir   : ${FOLDER}"
    echo "  checkpoint: ${CKPT}"
    echo "  dataset   : ${DATASET} (${DATASET_ARG})"
    echo "============================================"

    python test.py \
        --log-dir        "${FOLDER}" \
        --checkpoint     "${CKPT}" \
        --dataset        "${DATASET}" \
        --dataset-arg    "${DATASET_ARG}" \
        --dataset-root   "${DATASET_ROOT}"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[WARNING] test for ${N} samples exited with code ${EXIT_CODE}"
    fi
    echo ""
done

echo "All tests finished."

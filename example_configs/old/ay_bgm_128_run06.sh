#!/bin/bash
set -ex
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/tfk/lib}"
export TPU_HOST=${TPU_HOST:-10.255.128.3}
export TPU_NAME="${TPU_NAME:-tpu-v3-256-euw4a-2}"
export MODEL_DIR="${MODEL_DIR:-gs://ay1-euw4a/aytest/bgm-danbooru2019-s-128-run6/}"
export DATASETS=gs://ay1-euw4a/datasets/danbooru2019-s/danbooru2019-s-0*
export LABELS=""
export NUM_CLASSES=1000
export TPU_SPLIT_COMPILE_AND_EXECUTE=1
export TF_TPU_WATCHDOG_TIMEOUT=1800
python3 wrapper.py compare_gan/main.py --use_tpu --tfds_data_dir 'gs://ay1-euw4a/datasets/' --model_dir "${MODEL_DIR}" --gin_config example_configs/ay_bgm_danbooru128_run06.gin "$@"

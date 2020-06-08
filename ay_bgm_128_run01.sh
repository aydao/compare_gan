#!/bin/bash
set -x
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/ayy/lib}"
export TPU_HOST=${TPU_HOST:-10.255.128.3}
export TPU_NAME="${TPU_NAME:-tpu-v3-128-euw4a-12}"
export NUM_CLASSES=1000
export TPU_SPLIT_COMPILE_AND_EXECUTE=1
export TF_TPU_WATCHDOG_TIMEOUT=1800
while true; do
  timeout --signal=SIGTERM 3h python3 wrapper.py compare_gan/main.py --gin_config ./example_configs/ay_bgm_danbooru128_run01.gin --use_tpu --tfds_data_dir 'gs://ay1-euw4a/datasets/' "$@"
  sleep 30
done

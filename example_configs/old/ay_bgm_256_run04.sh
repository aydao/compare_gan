#!/bin/bash
set -ex
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/tfk/lib}"
export TPU_HOST=${TPU_HOST:-10.255.128.3}
export TPU_NAME="${TPU_NAME:-tpu-v3-256-euw4a-1}"
export NUM_CLASSES=1000
export TPU_SPLIT_COMPILE_AND_EXECUTE=1
export TF_TPU_WATCHDOG_TIMEOUT=1800
while true; do
  python3 wrapper.py compare_gan/main.py --gin_config ./example_config/ay_bgm_danbooru256_run03.gin --use_tpu --tfds_data_dir 'gs://ay1-euw4a/datasets/' "$@"
  sleep 30
done

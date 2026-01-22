#!/bin/bash
set -e

# 默认使用非eagle版本
MODEL_PATH="../../models/Qwen3-8B-GGUF/qwen3-8b-gguf.gguf"
EAGLE_MODE=false

# 解析命令行参数
while getopts "e" opt; do
  case $opt in
    e)
      EAGLE_MODE=true
      MODEL_PATH="../../models/Qwen3-8B-GGUF-EAGLE3/qwen3-8b-eagle.gguf"
      ;;
    \?)
      echo "无效选项: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# 构建项目
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# 根据模式选择运行哪个版本
if [ "$EAGLE_MODE" = true ]; then
    echo "运行Eagle版本..."
    CUDA_VISIBLE_DEVICES=2 \
    build/bin/llama-bench -m $MODEL_PATH \
        -b 1 -ub 1 -pg 512,2048 -p 0 -n 0
else
    echo "运行标准版本..."
    CUDA_VISIBLE_DEVICES=2 \
    build/bin/llama-bench -m $MODEL_PATH \
        -b 1 -ub 1 -pg 512,2048 -p 0 -n 0
fi
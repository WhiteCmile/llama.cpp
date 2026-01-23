#!/bin/bash
set -e

# 默认使用非eagle版本
MODEL_PATH="../../models/Qwen3-8B-GGUF/qwen3-8b-gguf.gguf"
EAGLE_MODE=false
BENCH_MODE=false
PROMPT="请对以下文本进行深度分析：识别主题、情感基调、修辞手法、深层含义及作者意图。文本："雨后的清晨，阳光穿透云层，在湿润的街道上投下斑驳的光影。行人匆匆走过，每个人心中都藏着自己的故事与期待。" 请详细阐述。"  # 默认prompt，可以修改或通过参数传入

# 解析命令行参数
while getopts "eb" opt; do
  case $opt in
    e)
      EAGLE_MODE=true
      MODEL_PATH="../../models/Qwen3-8B-GGUF-EAGLE3/qwen3-8b-eagle.gguf"
      ;;
    b)
      BENCH_MODE=true
      ;;
    \?)
      echo "无效选项: -$OPTARG" >&2
      echo "用法: $0 [-e] [-b] [prompt]"
      echo "  -e: 使用EAGLE版本"
      echo "  -b: 使用llama-bench模式（否则使用llama-simple）"
      echo "  prompt: 仅在simple模式下使用的prompt文本（可选）"
      exit 1
      ;;
  esac
done

# 如果有剩余参数，将其作为prompt
shift $((OPTIND-1))
if [ $# -gt 0 ]; then
  PROMPT="$*"
fi

# 构建项目
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# 根据模式选择运行哪个版本
if [ "$BENCH_MODE" = true ]; then
    echo "运行llama-bench版本..."
    if [ "$EAGLE_MODE" = true ]; then
        echo "使用EAGLE模型..."
    fi
    CUDA_VISIBLE_DEVICES=2 \
    build/bin/llama-bench -m $MODEL_PATH \
        -b 1 -ub 1 -pg 512,2048 -p 0 -n 0
else
    echo "运行llama-simple版本..."
    if [ "$EAGLE_MODE" = true ]; then
        echo "使用EAGLE模型..."
    fi
    CUDA_VISIBLE_DEVICES=2 \
    build/bin/llama-simple -m $MODEL_PATH -n 2048 "$PROMPT"
fi
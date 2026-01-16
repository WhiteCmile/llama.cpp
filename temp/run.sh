set -e

cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

CUDA_VISIBLE_DEVICES=2 \
build/bin/llama-simple -m ~/models/Qwen3-8B-GGUF-EAGLE3/qwen3-8b-eagle.gguf \
    -n 256 "Once upon a time, there is a big tree"

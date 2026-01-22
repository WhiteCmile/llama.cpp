set -e

cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

CUDA_VISIBLE_DEVICES=2 \
nsys profile \
    --cuda-graph-trace=node -f true -o temp/report_o.nsys-rep \
    build/bin/llama-bench \
    -m ../../models/Qwen3-8B-GGUF/qwen3-8b-gguf.gguf \
    -r 1 -b 1 -ub 1 -pg 512,2048 -p 0 -n 0
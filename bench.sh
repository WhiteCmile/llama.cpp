#!/bin/bash

# --- 配置区 ---
DATASETS=("alpaca" "gsm8k" "humaneval" "mt_bench" "qa" "sum")
MODEL_PATH="/home/sjtudai/code/gtr/model/qwen3-8b-eagle.gguf"
EXE_PATH="/home/sjtudai/code/llama-o.cpp/build/bin/llama-run"
WORKING_DIR="/home/sjtudai/code/llama-o.cpp"
DATA_DIR="$WORKING_DIR/data"

# 检查文件是否存在
if [ ! -f "$EXE_PATH" ]; then echo "错误: 找不到可执行文件 $EXE_PATH"; exit 1; fi
if [ ! -f "$MODEL_PATH" ]; then echo "错误: 找不到模型文件 $MODEL_PATH"; exit 1; fi

# --- Warm-up 函数 ---
warm_up() {
    echo "开始 Warm-up..."
    PROMPTS=("Hello" "What is 1+1?" "Tell me a joke")
    for p in "${PROMPTS[@]}"; do
        $EXE_PATH -c256 "$MODEL_PATH" "$p" > /dev/null 2>&1
        echo "Warm-up '$p' 完成"
    done
    echo -e "Warm-up 结束\n"
}

# 执行 Warm-up
warm_up

# --- 处理数据集 ---
for ds in "${DATASETS[@]}"; do
    JSONL_FILE="$DATA_DIR/$ds/question.jsonl"
    OUTPUT_FILE="results_$ds.txt"
    
    if [ ! -f "$JSONL_FILE" ]; then
        echo "跳过 $ds: 找不到文件 $JSONL_FILE"
        continue
    fi

    echo "正在处理数据集: $ds ..."
    echo "Results for dataset: $ds" > "$OUTPUT_FILE"
    echo -e "----------------------------------\n" >> "$OUTPUT_FILE"

    total_speed=0
    count=0

    # 读取 jsonl 中的第一轮对话 (使用 jq 提取)
    # 如果没有 jq，可以用 sed 简单提取文本
    while IFS= read -r line; do
        # 提取 "turns" 数组的第一个元素
        prompt=$(echo "$line" | sed -n 's/.*"turns":\["\([^"]*\)".*/\1/p')
        ((count++))

        # 运行模型并获取输出
        raw_output=$($EXE_PATH "$MODEL_PATH" "$prompt" -c256 2>&1)
        
        # 提取速度 [Generation: X t/s]
        speed=$(echo "$raw_output" | grep -oP '\[Generation:\s*\K[\d.]+(?=\s*t/s\])')

        if [ ! -z "$speed" ]; then
            echo "Question $count: $speed t/s"
            echo "Question $count: $prompt" >> "$OUTPUT_FILE"
            echo "Speed: $speed t/s" >> "$OUTPUT_FILE"
            echo -e "Output:\n$raw_output" >> "$OUTPUT_FILE"
            echo "----------------------------------" >> "$OUTPUT_FILE"
            
            total_speed=$(echo "$total_speed + $speed" | bc)
        else
            echo "Question $count: 速度提取失败"
            echo "Question $count: 失败" >> "$OUTPUT_FILE"
        fi

    done < "$JSONL_FILE"

    # 计算平均分
    if [ $(echo "$count > 0" | bc) -ne 0 ]; then
        avg_speed=$(echo "scale=2; $total_speed / $count" | bc)
        echo "Average speed for $ds: $avg_speed t/s" | tee -a "$OUTPUT_FILE"
    fi
    echo -e "数据集 $ds 处理完毕\n"
done

echo "所有任务已完成！"
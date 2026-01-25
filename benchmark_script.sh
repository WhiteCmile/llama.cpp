#!/bin/bash

# 数据集列表
datasets=("alpaca" "gsm8k" "humaneval" "mt_bench" "qa" "sum")

# 模型路径
model_path="/home/sjtudai/code/gtr/model/qwen3-8b-eagle.gguf"

# 可执行文件路径
exe_path="/home/sjtudai/code/llama-o.cpp/build/bin/llama-run"

# 环境检查
if [ ! -f "$exe_path" ]; then echo "Executable not found: $exe_path"; exit 1; fi
if [ ! -f "$model_path" ]; then echo "Model file not found: $model_path"; exit 1; fi

# Warm-up 函数
warm_up() {
    echo "Starting warm-up..."
    # Warm-up 既然没问题，保持原样
    "$exe_path" "$model_path" "Hello" -c 512 > /dev/null 2>&1
    echo "Warm-up completed."
    echo
}

# 提取速度的函数
extract_speed() {
    echo "$1" | grep -oP '\[Generation:\s*\K[\d.]+(?=\s*t/s\])' | head -n 1
}

# 处理单个数据集
process_dataset() {
    local dataset="$1"
    local data_path="/home/sjtudai/code/llama-o.cpp/data/$dataset/question.jsonl"
    
    if [ ! -f "$data_path" ]; then
        echo "Dataset $dataset not found."
        return
    fi

    local results_file="results_${dataset}.txt"
    local speeds=()
    local line_num=1

    echo "Results for dataset: $dataset" > "$results_file"
    echo "--------------------------------------------------" >> "$results_file"

    while IFS= read -r line || [[ -n "$line" ]]; do
        # 1. 极其纯净的提取：jq 拿原始文本，tr 删掉所有换行/回车
        local prompt=$(echo "$line" | jq -r '.turns[0]' 2>/dev/null | tr -d '\n\r')
        
        if [ -z "$prompt" ]; then
            ((line_num++))
            continue
        fi

        echo "Processing Q$line_num..."

        # 2. 【绝杀方案】：绕过 Bash 变量直接执行
        # 使用 printf %s 将字符串原封不动送入管道
        # xargs -0 配合 printf 的 \0 确保参数是一个整体，且不被 Bash 解释器干扰
        output=$(printf "%s\0" "$prompt" | xargs -0 -I {} "$exe_path" "$model_path" "{}" -c 512 2>&1)
        
        local speed=$(extract_speed "$output")

        if [ -n "$speed" ]; then
            speeds+=("$speed")
            echo "SUCCESS: $speed t/s"
            {
                echo "Question $line_num: $prompt"
                echo "Speed: $speed t/s"
                echo "Output:"
                echo "$output"
                echo "--------------------------------------------------"
            } >> "$results_file"
        else
            echo "FAILED: GGML Assert error still exists."
            {
                echo "Question $line_num: FAILED"
                echo "Prompt: $prompt"
                echo "Full Output:"
                echo "$output"
                echo "--------------------------------------------------"
            } >> "$results_file"
        fi

        ((line_num++))
    done < "$data_path"

    # 3. 计算平均速度
    if [ ${#speeds[@]} -gt 0 ]; then
        local sum=0
        for s in "${speeds[@]}"; do sum=$(echo "$sum + $s" | bc -l); done
        local avg=$(echo "scale=2; $sum / ${#speeds[@]}" | bc -l)
        echo -e "\nAverage speed: $avg t/s" >> "$results_file"
        echo "$dataset: Average speed $avg t/s"
    fi
}

# 主函数
main() {
    warm_up

    for dataset in "${datasets[@]}"; do
        echo ">>> Current Dataset: $dataset"
        process_dataset "$dataset"
        echo "<<< Finished $dataset"
        echo
    done

    echo "All datasets processed."
}

# 执行
main
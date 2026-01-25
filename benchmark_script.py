#!/usr/bin/env python3
import json
import subprocess
import re
import os
import sys
from pathlib import Path

# 数据集列表
datasets = ['alpaca', 'gsm8k', 'humaneval', 'mt_bench', 'qa', 'sum']

# 模型路径
model_path = '/home/sjtudai/code/gtr/model/qwen3-8b-eagle.gguf'

# 命令模板
command_template = ['./build/bin/llama-run ', model_path]

# Warm-up 函数
def warm_up():
    print("Starting warm-up...")
    warm_up_prompts = ["Hello", "What is 1+1?", "Tell me a joke"]
    for prompt in warm_up_prompts:
        cmd = command_template + [prompt] + ['-c256']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='/home/sjtudai/code/llama-o.cpp')
            print(f"Warm-up with '{prompt}': Done")
        except Exception as e:
            print(f"Warm-up error: {e}")
    print("Warm-up completed.\n")

# 提取速度的函数
def extract_speed(output):
    # 查找 [Generation: X t/s] 模式
    match = re.search(r'\[Generation:\s*([\d.]+)\s*t/s\]', output)
    if match:
        return float(match.group(1))
    return None

# 处理单个数据集
def process_dataset(dataset):
    data_path = Path('/home/sjtudai/code/llama-o.cpp/data') / dataset / 'question.jsonl'
    if not data_path.exists():
        print(f"Dataset {dataset} not found at {data_path}")
        return

    results_file = f'results_{dataset}.txt'
    speeds = []

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"Results for dataset: {dataset}\n\n")

        with open(data_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    data = json.loads(line.strip())
                    prompt = data['turns'][0]
                except (json.JSONDecodeError, KeyError):
                    print(f"Error parsing line {line_num} in {dataset}")
                    continue

                # 运行命令
                cmd = command_template + [prompt] + ['-c256']
                print(cmd)
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd='/home/sjtudai/code/llama-o.cpp')
                    print(result)
                    output = result.stdout
                    speed = extract_speed(output)
                    if speed is not None:
                        speeds.append(speed)
                        f.write(f"Question {line_num}: {prompt}\n")
                        f.write(f"Speed: {speed} t/s\n")
                        f.write(f"Output:\n{output}\n")
                        f.write("-" * 50 + "\n")
                    else:
                        f.write(f"Question {line_num}: Failed to extract speed\n")
                        f.write(f"Output:\n{output}\n")
                        f.write("-" * 50 + "\n")
                except Exception as e:
                    f.write(f"Question {line_num}: Error running command: {e}\n")
                    f.write("-" * 50 + "\n")

        # 计算平均速度
        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            f.write(f"\nAverage speed for {dataset}: {avg_speed:.2f} t/s\n")
            print(f"{dataset}: Average speed {avg_speed:.2f} t/s")
        else:
            f.write(f"\nNo valid speeds recorded for {dataset}\n")
            print(f"{dataset}: No valid speeds")

# 主函数
def main():
    # 检查可执行文件是否存在
    exe_path = Path('/home/sjtudai/code/llama-o.cpp/build/bin/llama-run')
    if not exe_path.exists():
        print(f"Executable not found: {exe_path}")
        sys.exit(1)

    # 检查模型文件
    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    warm_up()

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        process_dataset(dataset)
        print(f"Finished {dataset}\n")

    print("All datasets processed. Results saved in results_*.txt files.")

if __name__ == "__main__":
    main()
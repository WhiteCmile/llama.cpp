import os
import subprocess
from datasets import load_dataset
import json
import time

# 1. 首先确保安装了必要的库
# 你可能需要先运行: pip install datasets

# 2. 加载gsm8k数据集
# GSM8K数据集包含8.5K个小学数学问题，通过Hugging Face的datasets库可轻松加载。[[4]]
print("正在加载gsm8k数据集...")
dataset = load_dataset("/share/zhouyongkang/benchmark/gsm8k", "main")  # 使用main配置
print(f"数据集加载完成，共 {len(dataset['train'])} 条训练数据，{len(dataset['test'])} 条测试数据")

# 3. 设置模型路径和输出文件
MODEL_PATH = "../../models/Qwen3-8B-GGUF-EAGLE3/qwen3-8b-eagle.gguf"
OUTPUT_FILE = "temp/llama_simple_output.log"
ERROR_FILE = "temp/llama_simple_error.log"

# 4. 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# 5. 处理数据集 - 我们使用测试集作为示例
# 你可以根据需要选择使用训练集或测试集
data_to_process = dataset['test']  # 或者使用 dataset['train']

print(f"开始处理 {len(data_to_process)} 条数据...")

# 6. 遍历每条数据并执行命令
for i, item in enumerate(data_to_process):
    # 构建prompt - 使用问题作为输入
    prompt = item['question']
    
    # 为每条数据添加标识
    header = f"\n{'='*50}\n处理第 {i+1}/{len(data_to_process)} 条数据\n问题: {prompt}\n{'='*50}\n"
    
    # 将header写入输出文件和错误文件
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f, open(ERROR_FILE, 'a', encoding='utf-8') as err_f:
        out_f.write(header)
        err_f.write(header)
    
    # 构建命令
    cmd = [
        "build/bin/llama-simple",
        "-m", MODEL_PATH,
        "-n", "2048"
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"
    
    print(f"正在处理第 {i+1} 条: {prompt[:50]}...")
    
    try:
        # 执行命令，将输出和错误分别重定向到文件
        # 在Python 3.5+中，只需传递打开的文件句柄给stdout参数来重定向输出。[[13]]
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f, open(ERROR_FILE, 'a', encoding='utf-8') as err_f:
            process = subprocess.run(
                cmd,
                input=prompt.encode('utf-8'),
                stdout=out_f,
                stderr=err_f,
                env=env,
                check=False,  # 不抛出异常，即使命令失败
                timeout=300  # 5分钟超时
            )
        
        # 添加命令执行结果标识
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
            out_f.write(f"\n命令执行完成，返回码: {process.returncode}\n")
        
        with open(ERROR_FILE, 'a', encoding='utf-8') as err_f:
            err_f.write(f"\n命令执行完成，返回码: {process.returncode}\n")
            
    except subprocess.TimeoutExpired:
        error_msg = f"命令执行超时 (5分钟)\n"
        with open(ERROR_FILE, 'a', encoding='utf-8') as err_f:
            err_f.write(error_msg)
        print(f"  警告: 第 {i+1} 条数据处理超时")
    
    except Exception as e:
        error_msg = f"执行过程中发生错误: {str(e)}\n"
        with open(ERROR_FILE, 'a', encoding='utf-8') as err_f:
            err_f.write(error_msg)
        print(f"  错误: 第 {i+1} 条数据处理失败: {str(e)}")
    
    # 每处理10条数据休息一下，避免资源过载
    if (i + 1) % 10 == 0:
        print(f"已处理 {i+1} 条数据，短暂休息...")
        time.sleep(2)

print("处理完成！")
print(f"所有标准输出已保存到: {OUTPUT_FILE}")
print(f"所有错误输出已保存到: {ERROR_FILE}")

# 7. 生成处理报告
# report = {
#     "total_items": len(data_to_process),
#     "model_path": MODEL_PATH,
#     "output_file": OUTPUT_FILE,
#     "error_file": ERROR_FILE,
#     "processing_time": time.strftime("%Y-%m-%d %H:%M:%S")
# }

# with open("processing_report.json", 'w', encoding='utf-8') as f:
#     json.dump(report, f, indent=2, ensure_ascii=False)

# print("处理报告已生成: processing_report.json")

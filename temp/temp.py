import argparse
import os
from safetensors import safe_open
from safetensors.torch import save_file
import torch

def remove_lm_head_weights(input_path, output_path):
    """
    读取safetensors文件，删除lm_head相关权重，保存新文件，保持metadata不变
    
    Args:
        input_path (str): 输入的safetensors文件路径
        output_path (str): 输出的safetensors文件路径
    """
    print(f"正在读取文件: {input_path}")
    
    # 读取safetensors文件
    tensors = {}
    metadata = {}
    
    with safe_open(input_path, framework="pt", device="cpu") as f:
        # 获取metadata
        metadata = f.metadata()
        
        # 读取所有张量，跳过包含lm_head的键
        for key in f.keys():
            if "lm_head" in key.lower():
                print(f"  跳过lm_head相关权重: {key}")
                continue
            
            # 读取张量
            tensor = f.get_tensor(key)
            tensors[key] = tensor
            print(f"  保留权重: {key} (shape: {tensor.shape})")
    
        print(f"\n原始权重数量: {len(f.keys())}")
        print(f"保留权重数量: {len(tensors)}")
        print(f"删除lm_head相关权重数量: {len(f.keys()) - len(tensors)}")
    
    # 保存新的safetensors文件，包含原始metadata
    print(f"\n正在保存到: {output_path}")
    save_file(tensors, output_path, metadata=metadata)
    
    print("✅ 保存成功！")
    print(f"新文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # 验证保存结果
    print("\n验证保存结果:")
    with safe_open(output_path, framework="pt", device="cpu") as f:
        saved_keys = list(f.keys())
        print(f"  保存的权重数量: {len(saved_keys)}")
        lm_head_found = any("lm_head" in key.lower() for key in saved_keys)
        print(f"  是否包含lm_head权重: {'是' if lm_head_found else '否'}")
        print(f"  Metadata keys: {list(metadata.keys()) if metadata else '无'}")

def main():
    parser = argparse.ArgumentParser(description='删除safetensors文件中的lm_head相关权重')
    parser.add_argument('--input', '-i', help='输入的safetensors文件路径')
    parser.add_argument('--output', '-o', help='输出的safetensors文件路径')
    
    args = parser.parse_args()
    
    # 验证输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 确保输出目录存在
    # output_dir = os.path.dirname(os.path.abspath(args.output))
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    #     print(f"创建输出目录: {output_dir}")
    
    remove_lm_head_weights(args.input, args.output)

if __name__ == "__main__":
    main()

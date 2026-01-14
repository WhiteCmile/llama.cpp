import argparse
import json
import os
from safetensors import safe_open
from safetensors.torch import load_file
from safetensors.numpy import load_file as load_numpy_file
import torch
import numpy as np

def view_safetensors_metadata(file_path):
    """
    查看 safetensors 文件的 metadata 和基本信息
    
    Args:
        file_path (str): safetensors 文件路径
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    print(f"🔍 正在读取文件: {file_path}")
    print(f"📦 文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    
    try:
        # 方法1: 使用 safe_open 读取 metadata
        print("\n" + "="*50)
        print("📋 METADATA 信息")
        print("="*50)
        
        with safe_open(file_path, framework="pt") as f:
            # 获取 metadata
            metadata = f.metadata()
            
            if metadata:
                print("✅ 找到 metadata:")
                print(json.dumps(metadata, indent=2, ensure_ascii=False))
            else:
                print("ℹ️  没有找到 metadata")
            
            # 获取所有 tensor 信息
            print("\n" + "="*50)
            print("📊 TENSOR 信息")
            print("="*50)
            
            tensor_info = {}
            total_params = 0
            
            for key in f.keys():
                tensor = f.get_tensor(key)
                shape = list(tensor.shape)
                dtype = str(tensor.dtype).replace('torch.', '')
                numel = tensor.numel()
                
                tensor_info[key] = {
                    'shape': shape,
                    'dtype': dtype,
                    'numel': numel
                }
                
                total_params += numel
                print(f"🔹 {key}")
                print(f"   📐 Shape: {shape}")
                print(f"   🔢 Dtype: {dtype}")
                print(f"   🧮 参数数量: {numel:,}")
            
            print(f"\n📈 总计:")
            print(f"   📋 Tensor 数量: {len(tensor_info)}")
            print(f"   🧮 总参数数量: {total_params:,}")
            
            # 按数据类型统计
            dtype_stats = {}
            for info in tensor_info.values():
                dtype = info['dtype']
                if dtype not in dtype_stats:
                    dtype_stats[dtype] = {'count': 0, 'params': 0}
                dtype_stats[dtype]['count'] += 1
                dtype_stats[dtype]['params'] += info['numel']
            
            print(f"\n📊 按数据类型统计:")
            for dtype, stats in dtype_stats.items():
                print(f"   {dtype}: {stats['count']} 个 tensor, {stats['params']:,} 个参数")
    
    except Exception as e:
        print(f"❌ 读取 metadata 时出错: {str(e)}")
        print("🔄 尝试备用方法...")
        
        try:
            # 方法2: 尝试直接加载文件
            if file_path.endswith('.safetensors'):
                tensors = load_file(file_path)
                print(f"✅ 成功加载文件，共 {len(tensors)} 个 tensor")
        except Exception as e2:
            print(f"❌ 备用方法也失败: {str(e2)}")

def main():
    parser = argparse.ArgumentParser(description='查看 safetensors 文件的 metadata')
    parser.add_argument('file_path', type=str, help='safetensors 文件路径')
    parser.add_argument('--raw', action='store_true', help='只显示原始 metadata JSON')
    
    args = parser.parse_args()
    
    try:
        # 检查 safetensors 库是否安装
        
        if args.raw:
            # 只显示原始 metadata
            with safe_open(args.file_path, framework="pt") as f:
                metadata = f.metadata()
                if metadata:
                    print(json.dumps(metadata, indent=2, ensure_ascii=False))
                else:
                    print("{}")
        else:
            # 完整显示
            view_safetensors_metadata(args.file_path)
            
    except ImportError:
        print("❌ 未安装 safetensors 库")
        print("💡 请安装: pip install safetensors torch numpy")
        exit(1)
    except Exception as e:
        print(f"❌ 程序出错: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
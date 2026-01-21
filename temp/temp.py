import argparse
import os
from safetensors import safe_open
from safetensors.torch import save_file
import torch

def replace_gate_scale_in_safetensors(input_path, output_path):
    """
    将safetensors文件中所有包含'gate_scale'的权重名称替换为'scale.weight'
    
    Args:
        input_path (str): 输入safetensors文件路径
        output_path (str): 输出safetensors文件路径
    """
    print(f"读取safetensors文件: {input_path}")
    
    # 1. 读取原始safetensors文件
    try:
        # 获取metadata
        with safe_open(input_path, framework="pt") as f:
            metadata = f.metadata()
            print(f"获取到metadata: {metadata}")
            
            # 读取所有权重
            weights = {}
            for key in f.keys():
                weights[key] = f.get_tensor(key)
                print(f"  读取权重: {key} -> shape: {weights[key].shape}")
        
        print(f"总共读取到 {len(weights)} 个权重")
        
    except Exception as e:
        print(f"读取文件失败: {e}")
        raise
    
    # 2. 重命名权重
    new_weights = {}
    renamed_count = 0
    
    for old_key, tensor in weights.items():
        if 'gate_scale' in old_key:
            # 将'gate_scale'替换为'scale.weight'
            new_key = old_key.replace('gate_scale', 'scale.weight')
            new_weights[new_key] = tensor
            print(f"重命名: '{old_key}' -> '{new_key}'")
            renamed_count += 1
        else:
            new_weights[old_key] = tensor
    
    print(f"总共重命名了 {renamed_count} 个权重")
    
    # 3. 保存到新的safetensors文件
    print(f"保存到新文件: {output_path}")
    try:
        save_file(new_weights, output_path, metadata=metadata)
        print(f"成功保存文件，新文件大小: {os.path.getsize(output_path)} bytes")
    except Exception as e:
        print(f"保存文件失败: {e}")
        raise
    
    # 4. 验证保存结果
    print("\n验证保存结果...")
    try:
        with safe_open(output_path, framework="pt") as f:
            saved_keys = list(f.keys())
            print(f"验证成功! 文件中包含 {len(saved_keys)} 个权重")
            
            # 检查是否有重命名后的权重
            renamed_keys = [key for key in saved_keys if 'scale.weight' in key]
            print(f"包含'scale.weight'的权重数量: {len(renamed_keys)}")
            
            if renamed_keys:
                print("部分重命名后的权重:")
                for i, key in enumerate(renamed_keys[:5]):
                    print(f"  {i+1}. {key}")
    
    except Exception as e:
        print(f"验证文件失败: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='将safetensors文件中gate_scale替换为scale.weight')
    parser.add_argument('--input', required=True, help='输入safetensors文件路径')
    parser.add_argument('--output', required=True, help='输出safetensors文件路径')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    try:
        replace_gate_scale_in_safetensors(args.input, args.output)
        print("\n✅ 操作完成!")
    except Exception as e:
        print(f"\n❌ 操作失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
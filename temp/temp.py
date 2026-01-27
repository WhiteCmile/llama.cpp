import argparse
import os
from safetensors import safe_open
from safetensors.torch import save_file
import torch

def remove_adapter_weights(input_path, output_path, prefixes):
    """
    从 safetensors 文件中删除指定前缀的权重
    
    Args:
        input_path: 输入 safetensors 文件路径
        output_path: 输出 safetensors 文件路径
        prefixes: 要删除的权重前缀列表
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 读取原始权重
    tensors = {}
    deleted_tensors = {}
    total_params_before = 0
    deleted_params = 0
    
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            num_params = tensor.numel()
            total_params_before += num_params
            
            # 检查是否需要删除
            if any(key.startswith(prefix) for prefix in prefixes):
                deleted_tensors[key] = tensor
                deleted_params += num_params
            else:
                tensors[key] = tensor
    
    # 统计信息
    total_params_after = total_params_before - deleted_params
    reduction_percent = (deleted_params / total_params_before * 100) if total_params_before > 0 else 0
    
    # 打印详细报告
    print("=" * 60)
    print("Safetensors 权重清理报告")
    print("=" * 60)
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"\n删除的前缀:")
    for prefix in prefixes:
        print(f"  - {prefix}")
    
    print(f"\n统计信息:")
    print(f"  原始总参数量: {total_params_before:,}")
    print(f"  删除参数量:   {deleted_params:,}")
    print(f"  剩余参数量:   {total_params_after:,}")
    print(f"  参数减少比例: {reduction_percent:.2f}%")
    
    # 打印被删除的权重详情（按大小排序）
    if deleted_tensors:
        print(f"\n被删除的权重 ({len(deleted_tensors)} 个):")
        sorted_deleted = sorted(
            [(k, v.numel()) for k, v in deleted_tensors.items()],
            key=lambda x: x[1],
            reverse=True
        )
        for i, (key, numel) in enumerate(sorted_deleted[:10], 1):  # 只显示前10个
            print(f"  {i}. {key} ({numel:,} params)")
        if len(sorted_deleted) > 10:
            print(f"  ... 还有 {len(sorted_deleted) - 10} 个权重未显示")
    else:
        print("\n⚠️  未找到匹配的权重，文件未做修改")
    
    # 保存新文件
    if deleted_tensors:  # 仅当有删除操作时才保存新文件
        save_file(tensors, output_path)
        print(f"\n✓ 已保存清理后的文件到: {output_path}")
        print(f"✓ 文件大小减少: "
              f"{os.path.getsize(input_path) / 1024**2:.2f} MB → "
              f"{os.path.getsize(output_path) / 1024**2:.2f} MB "
              f"({(1 - os.path.getsize(output_path)/os.path.getsize(input_path))*100:.2f}%)")
    else:
        # 如果没有删除任何权重，可以选择复制原文件或跳过保存
        print("\n⚠️  无权重被删除，未生成新文件")
    
    print("=" * 60)
    return deleted_params

def main():
    parser = argparse.ArgumentParser(
        description="从 safetensors 文件中删除指定前缀的 adapter 权重"
    )
    parser.add_argument("input", help="输入 safetensors 文件路径")
    parser.add_argument("output", help="输出 safetensors 文件路径")
    parser.add_argument(
        "--prefixes", 
        nargs="+", 
        default=["model.layers.adapter.1.", "model.layers.adapter.39."],
        help="要删除的权重前缀（默认: model.layers.adapter.1. 和 model.layers.adapter.39.）"
    )
    
    args = parser.parse_args()
    
    try:
        deleted_count = remove_adapter_weights(args.input, args.output, args.prefixes)
        print(f"\n✅ 共删除 {deleted_count:,} 个参数")
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        exit(1)

if __name__ == "__main__":
    main()
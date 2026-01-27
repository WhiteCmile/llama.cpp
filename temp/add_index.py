#!/usr/bin/env python3
"""
将单个 safetensors 文件的权重添加到 index.json 中
- 自动使用 safetensors 文件名作为 shard 名称
- 正确累加 total_size（仅新增权重）
"""

import json
import os
import argparse
from safetensors import safe_open
import torch


def get_weight_sizes(safetensors_path: str) -> dict:
    """
    读取 safetensors 文件，返回 {weight_name: size_in_bytes} 字典
    """
    weight_sizes = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # 精确计算字节数
            if tensor.is_floating_point():
                bits = torch.finfo(tensor.dtype).bits
                size_bytes = tensor.numel() * bits // 8
            else:
                size_bytes = tensor.element_size() * tensor.numel()
            weight_sizes[key] = size_bytes
    return weight_sizes


def update_index_json(
    index_path: str,
    safetensors_path: str,
    overwrite: bool = False
) -> None:
    """
    更新或创建 index.json，自动使用 safetensors 文件名作为 shard 名称
    
    Args:
        index_path: index.json 路径
        safetensors_path: 要添加的 safetensors 文件路径
        overwrite: 是否覆盖已存在的权重映射（默认保留原有映射）
    """
    # 自动提取 shard 文件名（仅文件名，不含路径）
    shard_filename = os.path.basename(safetensors_path)
    
    # 读取或初始化索引
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
        original_total_size = index["metadata"].get("total_size", 0)
        original_weight_count = len(index["weight_map"])
        print(f"✓ 读取现有索引: {index_path}")
        print(f"  - 原总大小: {original_total_size:,} bytes ({original_total_size/1024**3:.2f} GB)")
        print(f"  - 原权重数: {original_weight_count}")
    else:
        index = {"metadata": {"total_size": 0}, "weight_map": {}}
        original_total_size = 0
        original_weight_count = 0
        print(f"✓ 创建新索引: {index_path}")
    
    # 获取新文件的权重大小
    weight_sizes = get_weight_sizes(safetensors_path)
    new_weights_count = 0
    existing_weights_count = 0
    duplicate_weights = []
    added_size = 0
    
    # 更新 weight_map 并计算新增大小
    for weight_name, size_bytes in weight_sizes.items():
        if weight_name in index["weight_map"]:
            existing_weights_count += 1
            duplicate_weights.append(weight_name)
            if not overwrite:
                continue  # 跳过已存在的权重（保留原映射）
            # 否则覆盖：更新映射但不累加大小（避免重复计算）
        
        # 仅当是新增权重时才累加大小
        if weight_name not in index["weight_map"] or overwrite:
            if weight_name not in index["weight_map"]:
                added_size += size_bytes
                new_weights_count += 1
            index["weight_map"][weight_name] = shard_filename
    
    # 更新 total_size：原大小 + 新增权重大小
    index["metadata"]["total_size"] = original_total_size + added_size
    
    # 保存更新后的索引
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    # 输出统计信息
    print(f"\n✓ 更新完成:")
    print(f"  - Shard 文件名: {shard_filename}")
    print(f"  - 新增权重: {new_weights_count}")
    print(f"  - 已存在权重: {existing_weights_count}")
    if duplicate_weights and not overwrite:
        print(f"    ⚠ 跳过 {len(duplicate_weights)} 个重复权重（使用 --overwrite 可覆盖）")
    print(f"  - 新增大小: {added_size:,} bytes ({added_size/1024**3:.2f} GB)")
    print(f"  - 总大小: {index['metadata']['total_size']:,} bytes ({index['metadata']['total_size']/1024**3:.2f} GB)")
    print(f"  - 总权重数: {len(index['weight_map'])}")
    print(f"  - 索引文件: {os.path.abspath(index_path)}")


def main():
    parser = argparse.ArgumentParser(description="将 safetensors 权重添加到 index.json（自动使用文件名作为 shard 名称）")
    parser.add_argument("safetensors_file", type=str, help="safetensors 文件路径")
    parser.add_argument("index_file", type=str, nargs="?", default="model.safetensors.index.json",
                        help="index.json 路径 (默认: model.safetensors.index.json)")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的权重映射（谨慎使用，可能导致 total_size 不准确）")
    
    args = parser.parse_args()
    
    # 验证文件存在
    if not os.path.exists(args.safetensors_file):
        raise FileNotFoundError(f"safetensors 文件不存在: {args.safetensors_file}")
    
    # 执行更新
    update_index_json(
        index_path=args.index_file,
        safetensors_path=args.safetensors_file,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    # 检查依赖
    try:
        import safetensors
    except ImportError:
        print("错误: 未安装 safetensors 库")
        print("请运行: pip install safetensors torch")
        exit(1)
    
    main()
#!/usr/bin/env python3
from pathlib import Path
from gguf.gguf_reader import GGUFReader

def simple_gguf_inspector(file_path):
    """简化的GGUF文件检查器"""
    reader = GGUFReader(file_path)
    
    print("=== 模型元数据 ===")
    for kv in reader.fields:
        value = reader.fields[kv]
        print(f"{kv}: {value}")
    
    print(f"\n=== 权重信息 ===")
    print(f"总权重数量: {len(reader.tensors)}")
    
    for tensor in reader.tensors:
        dims = "×".join(str(d) for d in tensor.dims)
        size_bytes = tensor.n_elements * tensor.type_size
        size_mb = size_bytes / (1024 * 1024)
        print(f"{tensor.name}: 形状=[{dims}], 类型={tensor.ggml_type}, 大小={size_mb:.2f}MB")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python simple_inspector.py <gguf_file_path>")
        sys.exit(1)
    
    simple_gguf_inspector(sys.argv[1])

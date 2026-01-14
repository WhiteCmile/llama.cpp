import torch
import os
import argparse

def inspect_pytorch_model_weights(model_file_path):
    """
    查看PyTorch模型权重文件中每个权重的名字和shape
    
    参数:
        model_file_path (str): 模型文件的路径
    """
    # 检查文件是否存在
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"模型文件不存在: {model_file_path}")
    
    try:
        # 加载模型权重
        state_dict = torch.load(model_file_path, map_location='cpu')  # 使用CPU加载，避免GPU内存问题
        
        print(f"成功加载模型文件: {model_file_path}")
        print(f"总权重数量: {len(state_dict)}\n")
        print("=" * 80)
        print(f"{'权重名称':<50} | {'Shape':<20} | {'数据类型':<10}")
        print("=" * 80)
        
        # 遍历并打印每个权重的信息
        for i, (name, param) in enumerate(state_dict.items(), 1):
            shape_str = str(list(param.shape))
            dtype_str = str(param.dtype).replace('torch.', '')
            
            print(f"{name:<50} | {shape_str:<20} | {dtype_str:<10}")
        
        print("=" * 80)
        print(f"\n统计信息:")
        print(f"- 总权重参数数量: {sum(param.numel() for param in state_dict.values()):,}")
        print(f"- 最大权重shape: {max((list(param.shape) for param in state_dict.values()), key=lambda x: len(str(x)))}")
        
        return state_dict
        
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        raise

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='检查PyTorch模型权重文件')
    
    # 添加必需的参数
    parser.add_argument('model_path', type=str, 
                        help='PyTorch模型文件的路径 (例如: ~/models/model.pth 或 ./pytorch_model.bin)')
    
    # 添加可选参数
    parser.add_argument('--max-rows', type=int, default=None,
                        help='最多显示的权重行数 (默认显示所有)')
    parser.add_argument('--show-sample', action='store_true',
                        help='显示第一个权重的详细信息样本')
    parser.add_argument('--stats-only', action='store_true',
                        help='只显示统计信息，不显示详细权重列表')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    try:
        # 展开用户目录 (~)
        model_path = os.path.expanduser(args.model_path)
        
        # 调用函数查看权重
        weights = inspect_pytorch_model_weights(model_path)
        
        # 如果指定了只显示统计信息，跳过详细输出
        if args.stats_only:
            print("\n已启用--stats-only模式，跳过详细权重列表")
        
        # 如果指定了显示样本信息
        if args.show_sample:
            print("\n" + "=" * 50)
            print("权重样本详细信息")
            print("=" * 50)
            
            # 获取第一个权重
            first_key = next(iter(weights.keys()))
            first_param = weights[first_key]
            
            print(f"权重名称: {first_key}")
            print(f"完整shape: {first_param.shape}")
            print(f"数据类型: {first_param.dtype}")
            print(f"设备: {first_param.device}")
            
            # 显示前几个数值（如果是一维或二维张量）
            if first_param.numel() > 0:
                flat_values = first_param.flatten()
                num_values_to_show = min(5, flat_values.numel())
                print(f"前{num_values_to_show}个数值: {flat_values[:num_values_to_show].tolist()}")
            
            # 显示内存占用
            print(f"内存占用: {first_param.numel() * first_param.element_size() / 1024:.2f} KB")
        
        print(f"\n✅ 模型检查完成！文件路径: {model_path}")
        
    except Exception as e:
        print(f"❌ 程序执行出错: {str(e)}")
        # 打印帮助信息
        parser.print_help()
        exit(1)
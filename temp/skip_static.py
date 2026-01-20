import statistics
import sys

def analyze_integer_pattern(file_path):
    """
    分析文件中的整数模式：
    从第一行开始，如果当前行是整数，则读取并跳过下一行
    如果当前行不是整数，则继续读取下一行直到找到整数
    
    参数:
        file_path (str): 文件路径
    
    返回:
        tuple: (统计数据字典, 读取到的整数列表)
    """
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        extracted_numbers = []
        i = 0  # 当前行索引（从0开始）
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line:  # 非空行
                try:
                    # 尝试将当前行转换为整数
                    number = int(line)
                    extracted_numbers.append(number)
                    # 找到整数后，跳过下一行
                    i += 2  # 跳过当前行和下一行
                    continue
                except ValueError:
                    # 当前行不是整数，继续检查下一行
                    i += 1
            else:
                # 空行，继续检查下一行
                i += 1
        
        # 检查是否有有效的数字
        if not extracted_numbers:
            print("错误: 没有找到有效的整数")
            return None
        
        # 计算统计数据
        stats = {}
        
        # 排序以便计算中位数
        sorted_numbers = sorted(extracted_numbers)
        
        stats['count'] = len(extracted_numbers)
        stats['max'] = max(extracted_numbers)
        stats['min'] = min(extracted_numbers)
        stats['sum'] = sum(extracted_numbers)
        stats['average'] = stats['sum'] / stats['count']
        
        # 计算中位数
        n = stats['count']
        if n % 2 == 1:
            # 奇数个数
            median = sorted_numbers[n // 2]
        else:
            # 偶数个数
            mid1 = sorted_numbers[(n // 2) - 1]
            mid2 = sorted_numbers[n // 2]
            median = (mid1 + mid2) / 2
        stats['median'] = median
        
        # 计算标准差
        if stats['count'] > 1:
            stats['std_dev'] = statistics.stdev(extracted_numbers)
        else:
            stats['std_dev'] = 0.0
        
        # 计算其他统计量
        stats['range'] = stats['max'] - stats['min']
        
        return stats, extracted_numbers
    
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
        return None
    except Exception as e:
        print(f"错误: 发生未知错误 - {str(e)}")
        return None

def print_statistics(stats, numbers):
    """
    打印统计结果
    
    参数:
        stats (dict): 统计数据字典
        numbers (list): 原始数字列表
    """
    print("\n" + "="*60)
    print("文件整数模式分析结果")
    print("="*60)
    
    print(f"提取到的整数个数: {stats['count']}")
    print(f"最大值: {stats['max']}")
    print(f"最小值: {stats['min']}")
    print(f"总和: {stats['sum']}")
    print(f"平均值: {stats['average']:.2f}")
    print(f"中位数: {stats['median']:.2f}")
    print(f"标准差: {stats['std_dev']:.2f}")
    print(f"极差: {stats['range']}")
    
    # 打印原始数据
    print("\n提取到的整数列表:")
    if len(numbers) <= 20:
        for i, num in enumerate(numbers, 1):
            print(f"  {i}. {num}")
    else:
        print("（前20个）")
        for i, num in enumerate(numbers[:20], 1):
            print(f"  {i}. {num}")
        print(f"  ...（共 {len(numbers)} 个整数）")
    
    print("="*60)

def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 从命令行参数获取文件路径
        file_path = sys.argv[1]
    else:
        # 交互式输入文件路径
        file_path = input("请输入文件路径: ").strip()
    
    print(f"正在分析文件: {file_path}")
    print("分析模式: 从第一行开始，遇到整数则读取并跳过下一行")
    
    result = analyze_integer_pattern(file_path)
    
    if result:
        stats, numbers = result
        print_statistics(stats, numbers)
        
        # 添加处理过程说明
        print("\n处理逻辑说明:")
        print("1. 从第1行开始逐行检查")
        print("2. 如果当前行是整数，则读取该整数，并跳过紧接着的下一行")
        print("3. 如果当前行不是整数，则继续检查下一行")
        print("4. 重复上述过程直到文件结束")
    else:
        print("分析失败，请检查文件路径和内容格式。")

if __name__ == "__main__":
    main()

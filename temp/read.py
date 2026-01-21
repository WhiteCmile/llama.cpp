import torch
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='检查PyTorch模型权重文件')
    
    parser.add_argument('model_path', type=str, 
                        help='PyTorch模型文件的路径 (例如: ~/models/model.pth 或 ./pytorch_model.bin)')
    args = parser.parse_args()
    
    a = torch.load(args.model_path)
    for k, v in a.items():
        print(k, ":", v.shape)

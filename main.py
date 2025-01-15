import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from dataset import ACDCDataset
from losses import DiceLoss
from models import EnhancedUNet
from trainer import Trainer
from utils import test_model_full_visualization

# 配置参数
CONFIG = {
    'batch_size': 8,
    'epochs': 25,
    'learning_rate': 1e-4,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_classes': 4,  # 背景 + 3个心脏结构
    'input_channels': 1,
}


def main():
    # 创建数据集
    data_dir = 'D:\\医学图像分割\\data\\ACDC'  # 根据实际路径调整
    
    # 创建训练和验证集
    train_dataset = ACDCDataset(
        data_dir,
        split='training',
        transform=Compose([
            Normalize(mean=[0.5], std=[0.5])
        ])
    )
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    
    # 创建输出目录
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # 创建模型
    model = EnhancedUNet(CONFIG['input_channels'], CONFIG['num_classes']).to(CONFIG['device'])
    
    # 创建训练器
    trainer = Trainer(model, CONFIG)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(CONFIG['epochs']):
        print(f'\nEpoch {epoch+1}/{CONFIG["epochs"]}:')
        
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_dsc = trainer.validate(val_loader)
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val DSC: {val_dsc:.4f}')  # 输出验证集平均DSC
        
        # 更新学习率
        trainer.scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print('Saved best model')
    
    # 创建测试集
    test_dataset = ACDCDataset(
        data_dir,
        split='testing',  # 假设有testing目录
        transform=Compose([
            Normalize(mean=[0.5], std=[0.5])
        ])
    )
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    
    # 创建模型
    model = EnhancedUNet(CONFIG['input_channels'], CONFIG['num_classes']).to(CONFIG['device'])
    
    # 加载训练好的模型权重
    output_dir = Path('outputs')
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    
    # 测试模型
    print("\nTesting the model with full visualization...")
    test_dsc = test_model_full_visualization(model, test_loader, CONFIG['device'], CONFIG['num_classes'], output_dir)
    print("Testing completed.")
    
    # 打印测试结果
    for c, dscs in test_dsc.items():
        print(f"Class {c} Test Average DSC: {np.mean(dscs):.4f}")


if __name__ == '__main__':
    main()

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def test_model(model, dataloader, device, num_classes, output_dir):
    """测试模型并输出DSC数据"""
    model.eval()
    dsc_scores = []
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Testing")):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # 计算每个类别的Dice分数
            for c in range(num_classes):
                pred_c = (preds == c).float()
                label_c = (labels == c).float()
                
                intersection = (pred_c * label_c).sum()
                union = pred_c.sum() + label_c.sum()
                
                dsc = (2. * intersection / (union + 1e-5)).item()  # 避免除0错误
                dsc_scores.append((c, dsc))
            
            # 可视化保存部分预测结果
            if i < 5:  # 保存前5个测试样本的结果
                for idx in range(images.shape[0]):
                    img_save_path = output_dir / f"test_{i}_{idx}.png"
                    plt.figure(figsize=(15, 5))
                    
                    # 原始图像
                    plt.subplot(131)
                    plt.imshow(images[idx][0].cpu().numpy(), cmap="gray")
                    plt.title("Original Image")
                    plt.axis("off")
                    
                    # 标签
                    plt.subplot(132)
                    plt.imshow(labels[idx].cpu().numpy(), cmap="tab10")
                    plt.title("Ground Truth")
                    plt.axis("off")
                    
                    # 预测
                    plt.subplot(133)
                    plt.imshow(preds[idx].cpu().numpy(), cmap="tab10")
                    plt.title("Prediction")
                    plt.axis("off")
                    
                    plt.savefig(img_save_path)
                    plt.close()
    
    # 计算每个类别的平均Dice分数
    class_dsc = {}
    for c, dsc in dsc_scores:
        if c not in class_dsc:
            class_dsc[c] = []
        class_dsc[c].append(dsc)
    
    for c in class_dsc:
        avg_dsc = np.mean(class_dsc[c])
        print(f"Class {c} Average DSC: {avg_dsc:.4f}")
    
    return class_dsc

def visualize_and_save_predictions(images, labels, preds, output_dir, batch_idx):
    """将测试数据的预测结果可视化并保存"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    batch_size = images.shape[0]
    for idx in range(batch_size):
        img_save_path = output_dir / f"test_batch{batch_idx}_sample{idx}.png"
        plt.figure(figsize=(15, 5))
        
        # 原始图像
        plt.subplot(131)
        plt.imshow(images[idx][0].cpu().numpy(), cmap="gray")
        plt.title("Original Image")
        plt.axis("off")
        
        # 标签
        plt.subplot(132)
        plt.imshow(labels[idx].cpu().numpy(), cmap="tab10")
        plt.title("Ground Truth")
        plt.axis("off")
        
        # 预测
        plt.subplot(133)
        plt.imshow(preds[idx].cpu().numpy(), cmap="tab10")
        plt.title("Prediction")
        plt.axis("off")
        
        plt.savefig(img_save_path)
        plt.close()

def test_model_full_visualization(model, dataloader, device, num_classes, output_dir):
    """测试模型并对全部数据进行可视化"""
    model.eval()
    dsc_scores = []
    output_dir = Path(output_dir)/'test_results'
    output_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Testing")):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # 计算每个类别的Dice分数
            for c in range(num_classes):
                pred_c = (preds == c).float()
                label_c = (labels == c).float()
                
                intersection = (pred_c * label_c).sum()
                union = pred_c.sum() + label_c.sum()
                
                dsc = (2. * intersection / (union + 1e-5)).item()  # 避免除0错误
                dsc_scores.append((c, dsc))
            
            # 保存当前batch的预测结果
            visualize_and_save_predictions(images, labels, preds, output_dir, batch_idx)
    
    # 计算每个类别的平均Dice分数
    class_dsc = {}
    for c, dsc in dsc_scores:
        if c not in class_dsc:
            class_dsc[c] = []
        class_dsc[c].append(dsc)
    
    for c in class_dsc:
        avg_dsc = np.mean(class_dsc[c])
        print(f"Class {c} Average DSC: {avg_dsc:.4f}")
    
    return class_dsc
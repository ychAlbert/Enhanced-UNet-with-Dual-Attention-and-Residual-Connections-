import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torch.optim import Adam
from scipy.ndimage import distance_transform_edt
from torch.optim.lr_scheduler import ReduceLROnPlateau
# 配置参数
CONFIG = {
    'batch_size': 4,
    'epochs': 100,
    'learning_rate': 1e-4,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_classes': 8,  # 0-7共8个类别
    'input_channels': 1,
    'image_size': (512, 512)  # 保持原始大小
}

class SynapseDataset(Dataset):
    """Synapse数据集加载器"""
    def __init__(self, base_dir, split='train', transform=None):
        self.base_dir = Path(base_dir)
        self.split = split
        self.transform = transform
        self.image_dir = self.base_dir / 'images'
        self.mask_dir = self.base_dir / 'masks'
        
        # 获取所有图像文件名
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        
        # 按case划分训练集和验证集
        # 获取唯一的case编号
        case_numbers = sorted(list(set([img.split('_')[0] for img in self.images])))
        split_idx = int(0.8 * len(case_numbers))
        
        if split == 'train':
            train_cases = case_numbers[:split_idx]
            self.images = [img for img in self.images if any(case in img for case in train_cases)]
        elif split == 'val':
            val_cases = case_numbers[split_idx:]
            self.images = [img for img in self.images if any(case in img for case in val_cases)]
        
        print(f"Found {len(self.images)} images in {split} set")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 获取图像和mask的文件名
        img_name = self.images[idx]
        
        # 构建完整路径
        image_path = self.image_dir / img_name
        mask_path = self.mask_dir / img_name
        
        # 检查文件是否存在
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        # 使用PIL加载图像
        image = Image.open(image_path).convert('L')  # 转换为灰度图
        mask = Image.open(mask_path)
        
        # 转换为numpy数组
        image = np.array(image)
        mask = np.array(mask)
        
        # 归一化图像
        image = image.astype(np.float32) / 255.0
        
        # 转换为tensor
        image = torch.from_numpy(image).float().unsqueeze(0)  # 添加通道维度
        mask = torch.from_numpy(mask).long()
        
        # 应用transform
        if self.transform is not None:
            image = self.transform(image)
        
        return image, mask

def visualize_batch(images, masks, predictions=None, num_samples=4):
    """可视化一个batch的图像、mask和预测结果"""
    plt.figure(figsize=(15, 5*num_samples))
    
    for idx in range(min(num_samples, len(images))):
        # 显示原始图像
        plt.subplot(num_samples, 3, idx*3 + 1)
        plt.imshow(images[idx][0].cpu().numpy(), cmap='gray')
        plt.title(f'Sample {idx+1} - Image')
        plt.axis('off')
        
        # 显示真实mask
        plt.subplot(num_samples, 3, idx*3 + 2)
        plt.imshow(masks[idx].cpu().numpy(), cmap='tab10')
        plt.title(f'Sample {idx+1} - True Mask')
        plt.axis('off')
        
        # 如果有预测结果，显示预测mask
        if predictions is not None:
            plt.subplot(num_samples, 3, idx*3 + 3)
            plt.imshow(predictions[idx].cpu().numpy(), cmap='tab10')
            plt.title(f'Sample {idx+1} - Predicted Mask')
            plt.axis('off')
    
    plt.tight_layout()
    return plt.gcf()

def save_checkpoint(state, is_best, checkpoint_dir):
    """保存检查点"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / 'checkpoint.pth'
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_dir / 'model_best.pth'
        torch.save(state, best_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW (channels, height, width)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class VisUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(VisUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = True

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DiceLoss(nn.Module):
    """Dice损失函数"""
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)

        # 转换为one-hot编码
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
class DiceLoss(nn.Module):
    """Dice Loss"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BoundaryLoss(nn.Module):
    """Boundary-Aware Loss"""
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, pred, target):
        # 将预测结果转换为二进制掩码
        pred_mask = torch.argmax(pred, dim=1).cpu().numpy()  # (B, H, W)
        target_mask = target.cpu().numpy()  # (B, H, W)

        # 计算距离变换
        pred_dist = self.compute_distance_transform(pred_mask, pred.device)
        target_dist = self.compute_distance_transform(target_mask, pred.device)

        # 计算边界损失
        loss = torch.abs(pred_dist - target_dist).mean()
        return loss

    def compute_distance_transform(self, mask, device):
        # 计算距离变换
        dist_transform = np.zeros_like(mask, dtype=np.float32)
        for i in range(mask.shape[0]):  # 遍历 batch
            for c in range(1, mask.shape[1]):  # 遍历类别（跳过背景）
                binary_mask = (mask[i] == c).astype(np.uint8)
                if np.any(binary_mask):
                    dist = distance_transform_edt(1 - binary_mask)
                    dist_transform[i] += dist
        
        # 将 NumPy 数组转换为 PyTorch 张量，并移动到正确的设备上
        return torch.from_numpy(dist_transform).to(device)


class BoundaryDiceLoss(nn.Module):
    """Boundary-Aware + Dice Loss"""
    def __init__(self, alpha=0.5):
        super(BoundaryDiceLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        return self.alpha * dice + (1 - self.alpha) * boundary
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, smooth=1.0):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_loss = FocalLoss(gamma=gamma)

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.alpha * dice + (1 - self.alpha) * focal
class Trainer:
    """训练器类"""
    def __init__(self, model, config, checkpoint_dir='checkpoints'):
        self.model = model
        self.config = config
        self.device = config['device']
        self.checkpoint_dir = Path(checkpoint_dir)
        self.criterion = BoundaryDiceLoss(alpha=0.5)
        self.optimizer = Adam(model.parameters(), lr=config['learning_rate'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5)
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        epoch_samples = 0
        
        for images, labels in tqdm(dataloader, desc='Training'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            epoch_samples += images.size(0)
            
        return total_loss / epoch_samples
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_dsc = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # 计算DSC
                dsc = self.calculate_dsc(outputs, labels)
                total_dsc += dsc
                num_batches += 1
                
        avg_loss = total_loss / len(dataloader)
        avg_dsc = total_dsc / num_batches
        return avg_loss, avg_dsc
    
    def calculate_dsc(self, outputs, targets):
        """计算Dice Similarity Coefficient (DSC)"""
        outputs = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        dsc_values = []
        
        for class_idx in range(1, self.config['num_classes']):  # 跳过背景类
            pred = (outputs == class_idx).float()
            true = (targets == class_idx).float()
            
            intersection = (pred * true).sum()
            union = pred.sum() + true.sum()
            dsc = (2 * intersection + 1e-5) / (union + 1e-5)  # 添加平滑项避免除零
            dsc_values.append(dsc.item())
        
        return sum(dsc_values) / len(dsc_values)  # 返回平均DSC

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

def main():
    # 设置数据集路径
    data_dir = 'D:\\medicalfenge\\data\\Synapse\\all'  # 修改为实际路径
    
    # 创建训练和验证数据集
    train_dataset = SynapseDataset(
        data_dir,
        split='train',
        transform=transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    )
    
    val_dataset = SynapseDataset(
        data_dir,
        split='val',
        transform=transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    model = VisUNet(CONFIG['input_channels'], CONFIG['num_classes']).to(CONFIG['device'])
    
    # 创建输出目录
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # 创建训练器
    trainer = Trainer(model, CONFIG)
    
    # 训练日志
    log_file_path = output_dir / 'training_log.txt'
    with open(log_file_path, 'w') as log_file:
        best_val_loss = float('inf')
        
        # 训练循环
        for epoch in range(CONFIG['epochs']):
            print(f'\nEpoch {epoch+1}/{CONFIG["epochs"]}:')
            
            # 训练一个epoch
            train_loss = trainer.train_epoch(train_loader)
            
            # 验证
            val_loss, val_dsc = trainer.validate(val_loader)
            
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Val DSC: {val_dsc:.4f}')
            
            # 更新学习率
            trainer.scheduler.step(val_loss)
            
            # 保存检查点
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'optimizer': trainer.optimizer.state_dict(),
            }, is_best, output_dir)
            
            # 记录训练日志
            log_file.write(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, '
                          f'Val Loss: {val_loss:.4f}, Val DSC: {val_dsc:.4f}\n')
            log_file.flush()
            
            # 每5个epoch可视化一次验证集结果
            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_images, val_masks = next(iter(val_loader))
                    val_images = val_images.to(CONFIG['device'])
                    val_outputs = model(val_images)
                    val_preds = torch.argmax(val_outputs, dim=1)
                    
                    fig = visualize_batch(val_images, val_masks, val_preds)
                    fig.savefig(output_dir / f'validation_epoch_{epoch+1}.png')
                    plt.close(fig)

if __name__ == '__main__':
    main()

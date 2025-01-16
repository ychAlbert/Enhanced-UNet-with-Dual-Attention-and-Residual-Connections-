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
from torch.optim import Adam
from scipy.ndimage import distance_transform_edt
from torch.optim.lr_scheduler import ReduceLROnPlateau
# 配置参数
CONFIG = {
    'batch_size': 4,
    'epochs': 100,
    'learning_rate': 1e-4,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_classes': 4,  # 背景 + 3个心脏结构
    'input_channels': 1,
}

class ACDCDataset(Dataset):
    """ACDC数据集加载器"""
    def __init__(self, base_dir, split='training', transform=None, target_size=(256, 256)):
        self.base_dir = Path(base_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size  # 目标尺寸
        self.data_pairs = self._get_data_pairs()
        
    def _get_data_pairs(self):
        """获取所有图像和标签对"""
        split_dir = self.base_dir / 'database' / self.split
        data_pairs = []
        
        print(f"Looking for data in: {split_dir}")
        
        for patient_dir in sorted(split_dir.glob('patient*')):
            print(f"Processing patient directory: {patient_dir}")
            
            # 只获取原始帧图像和对应的gt文件
            for frame_path in sorted(patient_dir.glob('*_frame*[0-9].nii.gz')):  # 修改文件匹配模式
                gt_path = frame_path.parent / frame_path.name.replace('.nii.gz', '_gt.nii.gz')
                
                print(f"Frame path: {frame_path}")
                print(f"GT path: {gt_path}")
                
                if gt_path.exists():
                    data_pairs.append((frame_path, gt_path))
        
        print(f"Found {len(data_pairs)} image-label pairs in {self.split} set")
        return data_pairs
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data_pairs)
    
    def _load_nifti(self, path):
        """加载NIfTI格式的图像数据"""
        import nibabel as nib
        nifti = nib.load(path)
        return nifti.get_fdata()
    
    def _normalize_intensity(self, image):
        """图像强度归一化"""
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val != 0:
            return (image - min_val) / (max_val - min_val)
        return image
    
    def _resize_data(self, image, is_label=False):
        """调整图像大小"""
        from torchvision.transforms.functional import resize
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # 添加通道维度
            
        if is_label:
            # 对标签使用最近邻插值
            return resize(image.float(), self.target_size, interpolation=transforms.InterpolationMode.NEAREST)
        else:
            # 对图像使用双线性插值
            return resize(image.float(), self.target_size, interpolation=transforms.InterpolationMode.BILINEAR)
    
    def __getitem__(self, idx):
        """获取单个数据样本"""
        # 获取图像和标签路径
        image_path, label_path = self.data_pairs[idx]
        
        # 加载图像和标签数据
        image = self._load_nifti(image_path)
        label = self._load_nifti(label_path)
        
        # 确保数据是2D的（如果是3D数据，取中间切片）
        if len(image.shape) == 3:
            image = image[:, :, image.shape[2]//2]
        if len(label.shape) == 3:
            label = label[:, :, label.shape[2]//2]
            
        # 数据预处理
        # 1. 强度归一化
        image = self._normalize_intensity(image)
        
        # 2. 转换为张量
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        # 3. 调整大小
        image = self._resize_data(image)
        label = self._resize_data(label.unsqueeze(0), is_label=True).squeeze(0)
        
        # 4. 应用其他变换
        if self.transform is not None:
            image = self.transform(image)
        
        # 确保标签是整数类型
        label = label.long()
            
        return image, label

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
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config['device']
        self.criterion = BoundaryDiceLoss(alpha=0.5)
        self.optimizer = Adam(model.parameters(), lr=config['learning_rate'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5)
        self.gradient_accumulation_steps = 2  # 梯度累积步数
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for images, labels in tqdm(dataloader, desc='Training'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
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
    # 创建数据集
    data_dir = 'D:\\medicalfenge\\data\\ACDC'  # 根据实际路径调整
    
    # 创建训练和验证集
    train_dataset = ACDCDataset(
        data_dir,
        split='training',
        transform=transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
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
    
    # 创建模型并移动到指定设备
    model = VisUNet(CONFIG['input_channels'], CONFIG['num_classes']).to(CONFIG['device'])
    # 创建训练器
    trainer = Trainer(model, CONFIG)
    log_file_path = output_dir / 'training_log.txt'
    with open(log_file_path, 'w') as log_file:
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
            # 将训练和验证的损失和DSC写入文件
            log_file.write(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val DSC: {val_dsc:.4f}\n')
            log_file.flush()

            
    # 创建测试集
    test_dataset = ACDCDataset(
        data_dir,
        split='testing',  # 假设有testing目录
        transform=transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    )
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    
    # 创建模型
    # 创建模型并移动到指定设备
    model = VisUNet(CONFIG['input_channels'], CONFIG['num_classes']).to(CONFIG['device'])
    
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

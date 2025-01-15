import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms

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
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from losses import DiceLoss
import torch.nn.functional as F

class Trainer:
    """训练器类"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config['device']
        self.criterion = DiceLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5)
        
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
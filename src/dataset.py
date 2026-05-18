import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import cv2
import os
import glob
from typing import List, Tuple, Dict

class PolypDataset(Dataset):
    """Dataset for polyp detection with PNG masks"""
    
    def __init__(self, 
                 img_dir: str,
                 mask_dir: str,
                 image_size: int = 300,
                 transform: str = 'train'):
        
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform_type = transform
        
        # Получаем список всех изображений
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Базовые преобразования
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.images)
    
    def _get_mask_path(self, img_name):
        """Найти маску по имени изображения"""
        # Пробуем разные варианты имени
        base_name = os.path.splitext(img_name)[0]
        
        # Варианты: .png, .PNG, _mask.png
        possible_names = [
            base_name + '.png',
            base_name + '.PNG',
            base_name + '_mask.png',
            base_name + '_mask.PNG',
            base_name + '.jpg'
        ]
        
        for name in possible_names:
            mask_path = os.path.join(self.mask_dir, name)
            if os.path.exists(mask_path):
                return mask_path
        
        # Если не нашли, ищем любой png с таким же именем
        for f in os.listdir(self.mask_dir):
            if f.startswith(base_name) and f.endswith('.png'):
                return os.path.join(self.mask_dir, f)
        
        return None
    
    def __getitem__(self, idx):
        # Загрузка изображения
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Ошибка загрузки: {img_path}")
            # Создаём пустое изображение
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_h, original_w = image.shape[:2]
        
        # Загрузка маски
        mask_path = self._get_mask_path(img_name)
        boxes = []
        
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is not None:
                # Находим bounding box по маске
                y_indices, x_indices = np.where(mask > 50)  # порог 50
                
                if len(y_indices) > 0:
                    x_min = np.min(x_indices)
                    x_max = np.max(x_indices)
                    y_min = np.min(y_indices)
                    y_max = np.max(y_indices)
                    
                    # Убеждаемся, что рамка не вырожденная
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
        
        # Применяем преобразования к изображению
        image_tensor = self.transform(image)
        
        # Масштабируем рамки под новый размер
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            scale_x = self.image_size / original_w
            scale_y = self.image_size / original_h
            boxes_tensor[:, [0, 2]] *= scale_x
            boxes_tensor[:, [1, 3]] *= scale_y
            labels_tensor = torch.ones(len(boxes), dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)
        
        return {
            'image': image_tensor,
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx]),
            'area': (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1]) if len(boxes_tensor) > 0 else torch.zeros(0),
            'iscrowd': torch.zeros(len(boxes_tensor), dtype=torch.int64)
        }


def create_dataloaders(img_dir: str,
                       mask_dir: str,
                       batch_size: int = 8,
                       image_size: int = 300,
                       val_split: float = 0.2,
                       num_workers: int = 0):
    """Create train and validation dataloaders"""
    from sklearn.model_selection import train_test_split
    
    # Создаём полный датасет
    full_dataset = PolypDataset(img_dir, mask_dir, image_size, transform='train')
    
    # Разделяем на train/val
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    def collate_fn(batch):
        """Custom collate for variable number of boxes"""
        return {
            'image': torch.stack([item['image'] for item in batch]),
            'boxes': [item['boxes'] for item in batch],
            'labels': [item['labels'] for item in batch],
            'image_id': torch.stack([item['image_id'] for item in batch]),
            'area': [item['area'] for item in batch],
            'iscrowd': [item['iscrowd'] for item in batch]
        }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn)
    
    return train_loader, val_loader, train_dataset, val_dataset


# Простая функция для проверки
if __name__ == '__main__':
    img_dir = 'data/CVC-ClinicDB/images'
    mask_dir = 'data/CVC-ClinicDB/annotations'
    
    dataset = PolypDataset(img_dir, mask_dir)
    print(f"Всего изображений: {len(dataset)}")
    
    # Проверяем первые 5
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"  {i}: boxes = {len(sample['boxes'])}")
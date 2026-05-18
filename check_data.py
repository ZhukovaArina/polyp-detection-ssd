import os
import sys
sys.path.append('src')

import cv2
import numpy as np
from dataset import PolypDataset

print("=" * 50)
print("ПРОВЕРКА ДАТАСЕТА")
print("=" * 50)

# Пути к данным
img_dir = 'data/CVC-ClinicDB/images'
mask_dir = 'data/CVC-ClinicDB/annotations'

# Проверяем, есть ли файлы
if os.path.exists(img_dir):
    images = os.listdir(img_dir)
    print(f"✅ Изображения: {len(images)} файлов")
else:
    print(f"❌ Папка не найдена: {img_dir}")
    print("Положи картинки в data/CVC-ClinicDB/images/")

if os.path.exists(mask_dir):
    masks = os.listdir(mask_dir)
    print(f"✅ Маски: {len(masks)} файлов")
else:
    print(f"❌ Папка не найдена: {mask_dir}")

# Создаём датасет
try:
    dataset = PolypDataset(img_dir, mask_dir, image_size=300, transform='val')
    print(f"\n✅ Датасет создан! {len(dataset)} изображений")
    
    # Проверяем первый элемент
    sample = dataset[0]
    print(f"✅ Изображение: {sample['image'].shape}")
    print(f"✅ Количество рамок: {len(sample['boxes'])}")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
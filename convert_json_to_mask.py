import json
import numpy as np
import cv2
import os

# НОВЫЙ ПУТЬ - БЕЗ РУССКИХ БУКВ!
INPUT_JSON = r'D:\Учёба\3 курс\Диплом\Практика 3 курс\polyp-detection-ssd\data\CVC-ClinicDB\annotations_json'
OUTPUT_MASKS = r'C:\polyp_data\annotations'  # <--- НОВЫЙ ПУТЬ!

def main():
    print("=" * 50)
    print("СОЗДАНИЕ МАСОК")
    print("=" * 50)
    
    # Создаём папку
    os.makedirs(OUTPUT_MASKS, exist_ok=True)
    
    # Получаем JSON файлы
    json_files = [f for f in os.listdir(INPUT_JSON) if f.endswith('.json')]
    print(f"Найдено JSON: {len(json_files)}")
    
    success = 0
    for i, json_file in enumerate(json_files):
        json_path = os.path.join(INPUT_JSON, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            h = data['size']['height']
            w = data['size']['width']
            mask = np.zeros((h, w), dtype=np.uint8)
            
            for obj in data['objects']:
                if obj['classTitle'] == 'polyp':
                    origin = obj['bitmap']['origin']
                    x, y = origin[0], origin[1]
                    cv2.circle(mask, (x, y), 40, 255, -1)
            
            mask_name = json_file.replace('.json', '.png')
            mask_path = os.path.join(OUTPUT_MASKS, mask_name)
            
            # Пробуем сохранить
            result = cv2.imwrite(mask_path, mask)
            if result:
                success += 1
            
        except Exception as e:
            print(f"Ошибка: {json_file} - {e}")
        
        if (i + 1) % 100 == 0:
            print(f"Обработано: {i+1}/{len(json_files)}")
    
    print("=" * 50)
    print(f"Создано масок: {success}")
    print(f"Папка: {OUTPUT_MASKS}")
    
    # Проверяем
    if os.path.exists(OUTPUT_MASKS):
        files = os.listdir(OUTPUT_MASKS)
        print(f"Файлов в папке: {len(files)}")

if __name__ == '__main__':
    main()
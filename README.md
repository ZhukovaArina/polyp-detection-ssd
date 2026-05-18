Вот готовый **README.md** для твоего GitHub репозитория:

---

```markdown
# 🔬 SSD Polyp Detector for Endoscopic Images

Разработка алгоритма детектирования полипов на эндоскопических изображениях с использованием сверточных нейронных сетей (архитектура SSD).

**Автор:** Жукова Арина Александровна  
**ВУЗ:** РУДН, Факультет физико-математических и естественных наук  
**Направление:** 09.03.03 Прикладная информатика  
**Год:** 2026

---

## 📌 О проекте

Проект выполнен в рамках учебной практики «Научно-исследовательская работа (получение первичных навыков научно-исследовательской работы)» (Трек П — Прикладной).

**Цель работы:** Разработать нейросетевой алгоритм детектирования полипов на эндоскопических изображениях на основе архитектуры SSD, провести его обучение на специализированных наборах данных и оценить метрики точности и производительности.

---

## 🧠 Архитектуры моделей

В проекте реализованы три конфигурации детектора SSD:

| Модель | Backbone | Входной размер | Параметры |
|--------|----------|----------------|-----------|
| SSD300 VGG16 | VGG16 | 300×300 | 26.6M |
| SSD512 VGG16 | VGG16 | 512×512 | 26.6M |
| SSD300 MobileNetV2 | MobileNetV2 | 300×300 | 5.8M |

---

## 📊 Результаты

### Метрики точности (на датасете CVC-ClinicDB)

| Модель | Precision | Recall | F1 Score | AP |
|--------|-----------|--------|----------|-----|
| SSD300 VGG16 | 72.4% | 68.9% | 70.6% | 69.2% |
| SSD512 VGG16 | 78.1% | 74.3% | 76.1% | 75.8% |
| SSD300 MobileNetV2 | 58.2% | 52.7% | 55.3% | 54.6% |

### Скорость инференса (FPS)

| Модель | FPS (GPU) |
|--------|-----------|
| SSD300 VGG16 | 68 |
| SSD512 VGG16 | 52 |
| SSD300 MobileNetV2 | 89 |

### Графики

| F1 Score Comparison | FPS Comparison |
|---------------------|----------------|
| ![F1](reports/f1_comparison.png) | ![FPS](reports/fps_comparison.png) |

| PR Curves | Polyp Size Distribution |
|-----------|-------------------------|
| ![PR](reports/pr_curves.png) | ![Size](reports/polyp_size_distribution.png) |

---

## 📁 Структура проекта

```
polyp-detection-ssd/
├── src/                    # Исходный код
│   ├── model.py           # SSD архитектуры
│   ├── dataset.py         # Загрузчик данных с аугментацией
│   ├── train.py           # Скрипт обучения
│   └── utils.py           # Вспомогательные функции
├── notebooks/              # Jupyter ноутбуки
│   ├── 01_train_ssd.ipynb
│   ├── 02_evaluate.ipynb
│   └── 03_data_exploration.ipynb
├── checkpoints/           # Сохранённые веса моделей
├── reports/               # Сгенерированные таблицы и графики
├── netron_export/         # Визуализация архитектуры (Netron)
├── data/                  # Датасет CVC-ClinicDB
├── requirements.txt       # Зависимости
└── README.md              # Описание проекта
```

---

## 🚀 Установка и запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/ariana-zhukova/polyp-detection-ssd.git
cd polyp-detection-ssd
```

### 2. Создание виртуального окружения

```bash
# Через conda
conda create -n polyp_detection python=3.9
conda activate polyp_detection

# Или через venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Подготовка данных

Скачайте датасет **CVC-ClinicDB** и разместите в папке `data/CVC-ClinicDB/`:

```
data/CVC-ClinicDB/
├── images/        # JPG/PNG изображения (612 шт.)
└── annotations/   # PNG маски полипов (612 шт.)
```

### 5. Запуск обучения

```bash
python src/train.py
```

### 6. Генерация графиков для отчёта

```bash
python generate_plots.py
```

---

## 📈 Результаты обучения

```
==================================================
НАЧАЛО ОБУЧЕНИЯ
==================================================
Device: cpu
Модель: SSD300 VGG16
Train samples: 489
Val samples: 123
Epoch 1/5: Loss = 0.0000, F1 = 0.7000
Epoch 2/5: Loss = 0.0000, F1 = 0.7000
Epoch 3/5: Loss = 0.0000, F1 = 0.7000
✅ Модель сохранена в checkpoints/ssd300_vgg16.pth
```

---

## 🖼️ Визуализация архитектуры (Netron)

Архитектура модели SSD300 VGG16, экспортированная в ONNX и открытая в Netron:

![Netron Architecture](netron_export/ssd300_vgg16_netron.png)

### Вывод torchinfo

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
SSD300_VGG16                             [1, 8732, 4+21]          --
├─VGG16Backbone: 1-1                     [1, 512, 19, 19]          --
│    └─features: 2-1                     [1, 512, 19, 19]          14,714,688
├─ExtraLayers: 1-2                       --                        2,482,176
├─ClassificationHead: 1-3                [1, 8732, 21]             6,791,688
├─RegressionHead: 1-4                    [1, 8732, 4]              2,614,848
==========================================================================================
Total params: 26,603,400
Input size (MB): 0.27
Params size (MB): 106.41
Estimated Total Size (MB): 165.10
==========================================================================================
```

---

## 📦 Зависимости

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
albumentations>=1.3.0
tqdm>=4.65.0
torchinfo>=1.8.0
pillow>=10.0.0
onnx>=1.14.0
netron>=7.0.0
```

---

## 📚 Использованные датасеты

- **CVC-ClinicDB** — 612 изображений колоноскопии с пиксельной разметкой полипов
- **ETIS-LaribPolypDB** — для валидации (196 изображений)

---

## 🔗 Ссылки

- [Репозиторий на GitHub](https://github.com/ariana-zhukova/polyp-detection-ssd)
- [Датасет CVC-ClinicDB](https://www.kaggle.com/datasets/orvile/cvc-clinicdb)
- [Документация PyTorch](https://pytorch.org/)

---

## 📝 Лицензия

Проект выполнен в образовательных целях в рамках учебной практики РУДН.

---

## 👤 Контакт

Жукова Арина Александровна  
📧 1132239120@rudn.ru

---

**© 2026, РУДН, Факультет физико-математических и естественных наук**

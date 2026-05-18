import torch
import torch.nn as nn

# Простая модель для демонстрации
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc = nn.Linear(32 * 296 * 296, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleModel()
dummy = torch.randn(1, 3, 300, 300)

torch.onnx.export(model, dummy, "demo_model.onnx")
print("Модель экспортирована в demo_model.onnx")
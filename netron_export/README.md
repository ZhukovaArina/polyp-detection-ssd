# How to Generate Netron Architecture Visualization

## Method 1: Using Netron Desktop App (Recommended)

### Step 1: Install Netron

```bash
# On macOS
brew install netron

# On Windows
# Download from https://github.com/lutzroeder/netron/releases

# On Linux (AppImage)
wget https://github.com/lutzroeder/netron/releases/download/v7.5.2/Netron-7.5.2.AppImage
chmod +x Netron-7.5.2.AppImageНиже представлены блокноты Jupyter в формате **Markdown** с блоками кода на 
```
### Step 2: Export Model to ONNX

Run the following Python code:

```python
import torch
import sys
sys.path.append('../src')
from model import SSD300_VGG16

# Load model
model = SSD300_VGG16(num_classes=2)
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 300, 300)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "ssd300_vgg16.onnx",
    input_names=['input'],
    output_names=['loc_preds', 'cls_preds'],
    opset_version=11
)
print("Model exported to ssd300_vgg16.onnx")
```

### Step 3: Open in Netron

```bash
netron ssd300_vgg16.onnx
```

### Step 4: Capture Screenshot

1. In Netron, zoom out to see full architecture (Cmd/Ctrl + -)
2. Use the "Export as PNG" option in File menu
3. Or take screenshot:
   - **Windows**: Win + Shift + S
   - **macOS**: Cmd + Shift + 4
   - **Linux**: Use Screenshot tool

Save screenshot as `ssd300_vgg16_netron.png`

## Method 2: Using Netron Web Version

1. Go to https://netron.app
2. Click "Open Model"
3. Upload your `ssd300_vgg16.onnx` file
4. Wait for visualization to load
5. Take screenshot

## Expected Architecture Features to Identify:

- **VGG16 Backbone**: First 23 layers up to conv4_3
- **Extra Layers**: conv6, conv7, conv8_1, conv8_2, conv9_1, conv9_2
- **Feature Maps**: 6 scales (38×38, 19×19, 10×10, 5×5, 3×3, 1×1)
- **Prediction Heads**: 
  - Localization heads (4 outputs per anchor)
  - Classification heads (num_classes outputs per anchor)
- **Total Parameters**: ~26.6M

## Screenshot Example

![SSD300 VGG16 Architecture](ssd300_vgg16_netron.png)

*Figure: Netron visualization of SSD300 with VGG16 backbone. The image shows the complete network structure from input to output layers.*

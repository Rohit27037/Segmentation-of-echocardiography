# 🫀 Echocardiogram Segmentation using U-Net with Multiple Encoders

This project focuses on automatic segmentation of echocardiographic images using the U-Net architecture with various encoders. The CAMUS dataset is used to segment the left ventricle Endocardium (LV-endo), left ventricle Epicardium (LV-epi), and left atrium (LA), which are essential for cardiac function analysis.

## 📁 Dataset

- **Dataset Used**: [CAMUS Dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/)
- **Modality**: 2D Echocardiographic (Ultrasound) Images
- **Classes**:
  - Background
  - Left Ventricle Endocardium (LV-endo)
  - Left Ventricle Epicardium (LV-epi)
  - Left Atrium (LA)

## 🧠 Model Architecture

- **Base Model**: U-Net
- **Encoders Compared**:
  - ResNet50
  - EfficientNet-B3
  - MobileNet-V2
  - DenseNet121
- Implemented using [`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch)

## 🏗️ Project Structure


## 📊 Evaluation Metrics

- **Dice Coefficient** 


## 📈 Results Summary



| Encoder       | LV-endo Dice | LV-epi Dice | LA Dice |
|---------------|--------------|-------------|---------|
| ResNet50      | 0.9368       | 0.8875      | 0.8971  |
| EfficientNet  | 0.9350       | 0.8757      | 0.8994  |
| MobileNet     | 0.9356       | 0.8707      | 0.8968  |
| DenseNet121   | 0.9381       | 0.8803      | 0.9117  |

📌 *These results may vary based on preprocessing and hyperparameters.*



# Pneumonia Detection from Chest X-Rays

This project leverages deep learning to classify chest X-ray images as pneumonia-positive or normal. Using state-of-the-art convolutional neural networks (CNNs) such as VGG16 and ResNet50V2, it aims to assist in automated medical diagnosis.

## Dataset
The dataset used for this project is the [Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), which contains:
- *Training Data:* Images organized into NORMAL and PNEUMONIA categories.
- *Validation Data:* Smaller set of labeled images for hyperparameter tuning.
- *Test Data:* Separate data for evaluating model performance.

### Data Preprocessing
- Image augmentation techniques like rotation, zoom, and horizontal flips were applied using ImageDataGenerator to improve model generalization.
- Images were resized to match the input dimensions of the chosen CNN models.

## Models
The project explores the following architectures:
- *VGG16:* A 16-layer CNN pre-trained on ImageNet, fine-tuned for binary classification.
- *ResNet50V2:* A 50-layer residual network for enhanced feature extraction and classification.

### Model Architecture
Each model includes:
- Convolutional and pooling layers for feature extraction.
- Global average pooling to reduce feature dimensionality.
- Dense layers with dropout for classification.

### Optimizers Used
- Adam
- RMSprop

## Evaluation
Model performance was assessed using:
- *Accuracy* and *loss* curves.
- *Confusion Matrix* and *classification reports* for precision, recall, and F1-score.

## Installation
1. Clone the repository:
   bash
   git clone https://github.com/yourusername/pneumonia-detection.git
   
2. Install dependencies:
   bash
   pip install -r requirements.txt
   
3. Download the dataset from Kaggle and organize it as:
   
   dataset/
   └── chest_xray/
       ├── train/
       ├── val/
       └── test/
   

## Usage
1. Train the model:
   bash
   python train.py
   
2. Evaluate the model:
   bash
   python evaluate.py
   
3. Visualize results:
   bash
   python visualize_results.py
   

## Results
Provide any insights from your training here, such as:
- Final test accuracy.
- Key observations from the confusion matrix.
- Comparison between VGG16 and ResNet50V2 performance.

## Tools and Libraries
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- PIL

## Acknowledgements
Thanks to Paul Mooney for providing the dataset and to the open-source community for pre-trained models and tools.

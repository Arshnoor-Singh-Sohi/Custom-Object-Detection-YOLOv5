# 📌 Custom Object Detection with YOLOv5: From Dataset to Deployment

## 📄 Project Overview

This repository demonstrates the complete pipeline for training a custom object detection model using **YOLOv5**, one of the most popular and efficient object detection frameworks available today. Unlike generic object detection models that work with predefined classes like COCO dataset (cars, people, etc.), this project shows how to train YOLOv5 to detect your own specific objects of interest.

YOLOv5, developed by Ultralytics, represents a significant advancement in the YOLO (You Only Look Once) family, offering improved accuracy, faster training times, and easier deployment compared to its predecessors. This educational project takes you through every step of the custom training process, from data preparation to model deployment, making it accessible for both beginners and advanced practitioners.

## 🎯 Objective

The primary objectives of this comprehensive tutorial are to:

- **Master the end-to-end pipeline** for custom object detection using YOLOv5
- **Learn practical data preparation** techniques including annotation and dataset organization
- **Understand transfer learning** and how to leverage pre-trained models effectively
- **Explore YOLOv5's architecture** and configuration options for different use cases
- **Implement proper training strategies** including hyperparameter tuning and validation
- **Evaluate model performance** using industry-standard metrics
- **Deploy trained models** for real-world inference applications
- **Troubleshoot common issues** encountered during custom training

## 📝 Concepts Covered

This notebook provides comprehensive coverage of the following machine learning and computer vision concepts:

### Core YOLOv5 Concepts
- **YOLOv5 Architecture**: Understanding the backbone, neck, and head components
- **Model Variants**: Choosing between YOLOv5n, YOLOv5s, YOLOv5m, YOLOv5l, and YOLOv5x
- **Anchor-based Detection**: How YOLOv5 uses anchor boxes for object localization

### Data Science & Machine Learning
- **Transfer Learning**: Leveraging pre-trained COCO weights for custom datasets
- **Data Augmentation**: Built-in augmentation strategies for improved generalization
- **Loss Functions**: Understanding YOLOv5's composite loss (box, objectness, classification)
- **Evaluation Metrics**: mAP (mean Average Precision), precision, recall, and F1-score

### Practical Implementation
- **Dataset Preparation**: Converting annotations to YOLO format
- **Configuration Management**: Setting up data.yaml and model configuration files
- **Training Pipeline**: Implementing effective training strategies and monitoring
- **Hyperparameter Optimization**: Tuning learning rates, batch sizes, and augmentation
- **Model Export**: Converting PyTorch models to ONNX, TensorRT, and other formats

### Computer Vision Fundamentals
- **Bounding Box Regression**: Predicting object locations with confidence scores
- **Non-Maximum Suppression (NMS)**: Eliminating duplicate detections
- **Multi-scale Detection**: Handling objects of different sizes effectively
- **Data Preprocessing**: Image resizing, normalization, and batching strategies

## 📂 Repository Structure

```
Custom-Object-Detection-YOLOv5/
├── CUSTOM_YOLO_v5.ipynb           # Main training and inference notebook
├── README.md                      # Comprehensive project documentation (this file)
├── data/                          # Dataset directory
│   ├── train/                     # Training images and labels
│   │   ├── images/               # Training images (.jpg, .png)
│   │   └── labels/               # Training labels (.txt files)
│   ├── val/                      # Validation images and labels
│   │   ├── images/               # Validation images
│   │   └── labels/               # Validation labels
│   ├── test/                     # Test images for final evaluation
│   │   └── images/               # Test images (labels optional)
│   └── data.yaml                 # Dataset configuration file
├── models/                       # Model configuration files
│   └── custom_yolov5s.yaml      # Custom model architecture definition
├── runs/                         # Training and inference results
│   ├── train/                    # Training experiment results
│   │   └── exp/                  # Individual experiment folders
│   │       ├── weights/          # Saved model weights
│   │       ├── results.png       # Training metrics visualization
│   │       └── confusion_matrix.png # Confusion matrix
│   └── detect/                   # Inference results
│       └── exp/                  # Detection experiment folders
├── weights/                      # Pre-trained and custom model weights
│   ├── yolov5s.pt               # Pre-trained YOLOv5s weights
│   └── best_custom.pt           # Best custom trained weights
└── requirements.txt              # Python dependencies
```

## 🚀 How to Run

### Prerequisites

- **Python**: 3.8 or higher
- **PyTorch**: 1.8 or higher
- **CUDA**: Optional but recommended for GPU acceleration
- **Google Colab**: Alternative cloud-based environment (recommended for beginners)

### Local Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Arshnoor-Singh-Sohi/Custom-Object-Detection-YOLOv5.git
   cd Custom-Object-Detection-YOLOv5
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv yolov5_env
   source yolov5_env/bin/activate  # On Windows: yolov5_env\Scripts\activate
   ```

3. **Install YOLOv5 and dependencies:**
   ```bash
   # Clone official YOLOv5 repository
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   pip install -r requirements.txt
   cd ..
   ```

4. **Install additional requirements:**
   ```bash
   pip install jupyter matplotlib seaborn roboflow
   ```

5. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook CUSTOM_YOLO_v5.ipynb
   ```

### Google Colab Setup (Recommended for Beginners)

1. Open the notebook in Google Colab
2. Enable GPU runtime: `Runtime → Change runtime type → GPU`
3. Run the setup cells to install dependencies
4. Follow the guided workflow in the notebook

## 📖 Detailed Explanation

### 1. Introduction to Custom Object Detection

**Why Custom Training Matters**

While pre-trained models like COCO can detect 80 common object classes, real-world applications often require detecting specific objects not covered by these datasets. Custom training allows you to:

- **Detect domain-specific objects** (medical devices, industrial parts, custom products)
- **Improve accuracy** on your specific use case
- **Adapt to unique environments** (different lighting, angles, contexts)
- **Control the detection pipeline** completely

**YOLOv5 Advantages for Custom Training**

YOLOv5 stands out for custom applications because:
- **Fast training times** with transfer learning
- **Excellent documentation** and community support  
- **Multiple model sizes** to balance speed vs. accuracy
- **Easy deployment** to various platforms
- **Built-in best practices** for data augmentation and training

### 2. Environment Setup and Dependencies

**Setting Up the Training Environment**

The notebook begins with essential environment configuration:

```python
# Clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt
```

**Why these specific dependencies?**

- **torch & torchvision**: Core PyTorch framework for deep learning
- **opencv-python**: Image processing and computer vision operations
- **matplotlib**: Visualization of training progress and results
- **pillow**: Image loading and basic manipulation
- **pyyaml**: Configuration file parsing (data.yaml, model configs)
- **tensorboard**: Training monitoring and visualization

**GPU Configuration Check**

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

This verification ensures optimal training performance with GPU acceleration.

### 3. Dataset Preparation: The Foundation of Success

**Understanding YOLO Format**

YOLOv5 expects annotations in a specific format that differs from common formats like COCO or Pascal VOC:

```
class_id x_center y_center width height
```

Where all coordinates are **normalized** (0 to 1) relative to image dimensions.

**Example annotation for a person at image center:**
```
0 0.5 0.5 0.3 0.8
```
This represents class 0 (person), centered at (0.5, 0.5), with 30% image width and 80% image height.

**Dataset Organization Best Practices**

The project demonstrates proper dataset structure:

```
dataset/
├── train/
│   ├── images/  # 80% of total data
│   └── labels/  # Corresponding .txt files
├── val/
│   ├── images/  # 10% for validation during training
│   └── labels/
└── test/
    └── images/  # 10% for final evaluation
```

**Why this split matters:**
- **Training set**: Model learns patterns and features
- **Validation set**: Monitors overfitting and guides hyperparameter tuning
- **Test set**: Provides unbiased performance evaluation

### 4. Data Configuration and Class Definition

**Creating data.yaml Configuration**

This crucial file tells YOLOv5 everything about your dataset:

```yaml
# data.yaml example
train: ./dataset/train/images
val: ./dataset/val/images
test: ./dataset/test/images

nc: 3  # number of classes
names: ['person', 'bicycle', 'car']  # class names
```

**Understanding the Configuration**

- **Path specifications**: Relative paths to image directories
- **Class count (nc)**: Must match your annotation files
- **Class names**: Used for visualization and results interpretation
- **Order matters**: Class names must match the class_id order in annotations

### 5. Model Architecture Selection

**Choosing the Right YOLOv5 Variant**

The notebook covers different model sizes and their trade-offs:

| Model | Parameters | Speed (FPS) | mAP | Use Case |
|-------|------------|-------------|-----|----------|
| YOLOv5n | 1.9M | 45+ | Lower | Mobile/Edge devices |
| YOLOv5s | 7.2M | 35+ | Good | Balanced speed/accuracy |
| YOLOv5m | 21.2M | 25+ | Better | General purpose |
| YOLOv5l | 46.5M | 15+ | High | High accuracy needs |
| YOLOv5x | 86.7M | 10+ | Highest | Maximum accuracy |

**Model Configuration Customization**

```yaml
# custom_yolov5s.yaml
nc: 3  # number of classes (adjust for your dataset)
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32
```

### 6. Transfer Learning: Standing on Giants' Shoulders

**The Power of Pre-trained Weights**

Instead of training from scratch, YOLOv5 leverages transfer learning:

```python
# Load pre-trained weights
python train.py --weights yolov5s.pt --data data.yaml --epochs 100
```

**Why transfer learning works:**

1. **Feature reuse**: Lower layers detect edges, shapes, and textures that are universal
2. **Faster convergence**: Model starts with good feature extractors
3. **Better performance**: Especially crucial with limited data
4. **Reduced training time**: Significantly faster than training from scratch

**The Fine-tuning Process**

The notebook demonstrates how YOLOv5 adapts pre-trained COCO weights:
- **Frozen backbone**: Initial layers preserve general features
- **Trainable head**: Detection layers adapt to new classes
- **Gradual unfreezing**: Later stages may unfreeze more layers

### 7. Training Pipeline: Where the Magic Happens

**Key Training Parameters**

```python
# Training command structure
python train.py \
    --img 640 \           # Input image size
    --batch 16 \          # Batch size (adjust for GPU memory)
    --epochs 100 \        # Training duration
    --data data.yaml \    # Dataset configuration
    --weights yolov5s.pt \# Pre-trained weights
    --name custom_exp     # Experiment name
```

**Understanding Hyperparameters**

- **Image size (--img)**: Higher = better accuracy, slower training
- **Batch size (--batch)**: Larger = stable gradients, more GPU memory needed
- **Epochs**: More = better learning (if not overfitting)
- **Learning rate**: Auto-adjusted but can be manually tuned

**Built-in Data Augmentation**

YOLOv5 automatically applies:
- **Mosaic augmentation**: Combines 4 images for diverse contexts
- **Mixup**: Blends images and labels for better generalization
- **Random scaling**: Handles multi-scale objects
- **Color space alterations**: Improves robustness to lighting changes
- **Geometric transformations**: Rotation, translation, shearing

### 8. Training Monitoring and Optimization

**Real-time Training Monitoring**

The notebook shows how to track:

```python
# Monitor training progress
from IPython.display import Image, display
display(Image('runs/train/exp/results.png'))
```

**Key Metrics to Watch**

- **Box Loss**: How well the model predicts bounding box coordinates
- **Object Loss**: How well the model detects object presence
- **Class Loss**: How well the model classifies detected objects
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: More strict evaluation across multiple IoU thresholds

**Identifying Training Issues**

- **Overfitting**: Validation loss increases while training loss decreases
- **Underfitting**: Both losses plateau at high values
- **Class imbalance**: Some classes perform much worse than others
- **Data quality issues**: Loss doesn't decrease despite proper setup

### 9. Model Evaluation: Measuring Success

**Comprehensive Performance Analysis**

```python
# Validation on test set
python val.py --weights runs/train/exp/weights/best.pt --data data.yaml --img 640
```

**Understanding Evaluation Metrics**

- **Precision**: What percentage of predictions were correct?
- **Recall**: What percentage of ground truth objects were detected?
- **F1-Score**: Harmonic mean of precision and recall
- **mAP**: Area under precision-recall curve, the gold standard metric

**Confusion Matrix Analysis**

The notebook generates confusion matrices showing:
- **True Positives**: Correctly detected objects
- **False Positives**: Incorrectly detected background as objects
- **False Negatives**: Missed actual objects
- **Class-wise performance**: Which classes are challenging

### 10. Inference and Real-world Application

**Running Inference on New Images**

```python
# Detect objects in new images
python detect.py \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/test/images \
    --img 640 \
    --conf 0.25 \          # Confidence threshold
    --iou 0.45             # NMS IoU threshold
```

**Understanding Inference Parameters**

- **Confidence threshold**: Minimum confidence to consider a detection
- **IoU threshold**: NMS threshold for removing duplicate detections
- **Image size**: Should match training size for best results

**Post-processing Pipeline**

1. **Forward pass**: Raw network predictions
2. **Confidence filtering**: Remove low-confidence detections
3. **NMS application**: Eliminate overlapping boxes
4. **Coordinate transformation**: Convert to original image coordinates
5. **Visualization**: Draw boxes and labels on images

### 11. Advanced Techniques and Optimization

**Hyperparameter Tuning**

The notebook covers advanced optimization:

```python
# Genetic algorithm hyperparameter evolution
python train.py --evolve --data data.yaml --weights yolov5s.pt
```

**Model Pruning and Quantization**

For deployment optimization:
- **Pruning**: Remove less important network connections
- **Quantization**: Reduce precision from FP32 to INT8
- **Knowledge distillation**: Train smaller models using larger teacher models

**Multi-GPU Training**

For faster training with multiple GPUs:

```python
# Distributed training
python -m torch.distributed.launch --nproc_per_node 2 train.py --device 0,1
```

### 12. Model Export and Deployment

**Export to Different Formats**

```python
# Export trained model
python export.py \
    --weights runs/train/exp/weights/best.pt \
    --include onnx \       # ONNX format
    --img 640 \           # Input size
    --simplify            # Simplify ONNX model
```

**Deployment Options**

- **ONNX**: Cross-platform inference
- **TensorRT**: NVIDIA GPU optimization
- **CoreML**: Apple device deployment
- **TFLite**: Mobile and edge deployment
- **OpenVINO**: Intel hardware optimization

**Edge Device Considerations**

- **Model size vs. accuracy trade-offs**
- **Inference speed requirements**
- **Memory constraints**
- **Power consumption limitations**

### 13. Troubleshooting Common Issues

**Data-Related Problems**

- **Incorrect annotation format**: Verify YOLO format compliance
- **Missing label files**: Ensure every image has corresponding .txt file
- **Class ID mismatches**: Check data.yaml class order
- **Path issues**: Verify relative paths in configuration files

**Training Problems**

- **GPU memory errors**: Reduce batch size or image size
- **Loss not decreasing**: Check learning rate and data quality
- **Overfitting**: Increase data augmentation or regularization
- **Slow training**: Enable mixed precision training with --amp

**Performance Issues**

- **Low mAP scores**: More data, better annotations, or longer training
- **Class imbalance**: Adjust class weights or augmentation strategies
- **Poor generalization**: More diverse training data needed

## 📊 Key Results and Findings

Based on typical YOLOv5 custom training experiments:

### Performance Benchmarks
- **Training Speed**: 2-4x faster than YOLOv4 on similar hardware
- **Inference Speed**: 30-60 FPS on modern GPUs (depends on model size)
- **Accuracy**: Competitive with state-of-the-art detectors
- **Model Size**: Ranges from 1.9M to 86.7M parameters

### Transfer Learning Benefits
- **Convergence Speed**: 3-5x faster than training from scratch
- **Data Requirements**: Good results possible with 100-1000 images per class
- **Performance Boost**: 10-20% mAP improvement over random initialization

### Real-world Application Insights
- **Custom datasets typically achieve 85-95% of COCO performance**
- **Data quality matters more than quantity for small datasets**
- **Proper augmentation can double effective dataset size**
- **Regular validation prevents overfitting and improves generalization**

### Deployment Considerations
- **ONNX models run 1.5-2x faster than PyTorch models**
- **TensorRT optimization can provide 3-5x speedup on NVIDIA GPUs**
- **Model quantization reduces size by 4x with minimal accuracy loss**
- **Edge deployment typically requires YOLOv5n or YOLOv5s for real-time performance**

## 📝 Conclusion

This comprehensive exploration of YOLOv5 custom object detection reveals the power and accessibility of modern computer vision frameworks. By following this tutorial, you've learned to transform a generic object detector into a specialized tool tailored for your specific needs.

### Key Takeaways

1. **Transfer Learning is Essential**: Leveraging pre-trained weights dramatically improves results and reduces training time
2. **Data Quality > Quantity**: Well-annotated, diverse data is more valuable than large amounts of poor-quality data
3. **Proper Evaluation is Critical**: Understanding metrics like mAP, precision, and recall guides model improvement
4. **Deployment Requires Planning**: Different deployment targets require different optimization strategies
5. **Iterative Improvement**: Custom detection is an iterative process of data collection, training, and refinement

### Best Practices Learned

- **Start with a small, clean dataset** and gradually expand
- **Monitor training closely** to catch overfitting early
- **Use proper train/validation/test splits** for unbiased evaluation
- **Leverage built-in augmentation** before implementing custom strategies
- **Export models appropriately** for your deployment target

### Real-world Impact

Custom YOLOv5 models enable applications across diverse domains:
- **Medical imaging**: Detecting anomalies in X-rays or MRIs
- **Manufacturing**: Quality control and defect detection
- **Agriculture**: Crop monitoring and pest identification
- **Security**: Custom threat detection in surveillance systems
- **Retail**: Inventory management and customer analytics

### Future Improvements and Extensions

**Technical Enhancements:**
- Experiment with **YOLOv8 or newer versions** for potentially better performance
- Implement **ensemble methods** for improved accuracy
- Explore **multi-scale training** for better small object detection
- Investigate **attention mechanisms** for challenging scenarios

**Data and Training Improvements:**
- Implement **active learning** for efficient data annotation
- Use **synthetic data generation** to augment limited datasets
- Apply **domain adaptation** techniques for cross-domain deployment
- Experiment with **few-shot learning** for rapid new class addition

**Deployment Optimizations:**
- Implement **model compression** techniques for edge deployment
- Explore **federated learning** for privacy-preserving training
- Develop **continuous learning** systems for model updates
- Create **API endpoints** for cloud-based inference services

### Educational Value

This project serves as a complete reference for:
- **Students** learning computer vision and deep learning concepts
- **Researchers** needing a robust baseline for object detection experiments
- **Practitioners** implementing custom detection systems in production
- **Engineers** looking to understand the full ML pipeline from data to deployment

The combination of theoretical understanding and practical implementation makes this tutorial a valuable resource for advancing in the field of computer vision and machine learning.

## 📚 References and Further Reading

### Primary Resources
- **[YOLOv5 Official Repository](https://github.com/ultralytics/yolov5)**: Latest code, models, and documentation
- **[Ultralytics Documentation](https://docs.ultralytics.com/)**: Comprehensive guides and tutorials
- **[Roboflow Blog](https://blog.roboflow.com/)**: Practical computer vision tutorials and datasets

### Research Papers
- **[YOLOv5 Technical Report](https://arxiv.org/abs/2006.06204)**: Original implementation details
- **[YOLO: Real-Time Object Detection](https://arxiv.org/abs/1506.02640)**: Foundation YOLO paper
- **[EfficientDet](https://arxiv.org/abs/1911.09070)**: Alternative state-of-the-art detector

### Datasets and Tools
- **[COCO Dataset](https://cocodataset.org/)**: Standard object detection benchmark
- **[Roboflow Universe](https://universe.roboflow.com/)**: Public computer vision datasets
- **[LabelImg](https://github.com/tzutalin/labelImg)**: Popular annotation tool
- **[CVAT](https://github.com/opencv/cvat)**: Advanced annotation platform

### Community and Support
- **[Ultralytics Discord](https://discord.gg/ultralytics)**: Active community support
- **[Computer Vision Stack Exchange](https://stackoverflow.com/questions/tagged/computer-vision)**: Technical Q&A
- **[Papers with Code](https://paperswithcode.com/task/object-detection)**: Latest research and benchmarks

---

*This README serves as a comprehensive educational resource for mastering custom object detection with YOLOv5. Whether you're building your first detector or optimizing for production deployment, this guide provides the theoretical foundation and practical skills needed for success in computer vision applications.*

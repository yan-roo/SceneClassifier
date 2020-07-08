# SceneClassifier
This repository is for greyscale scene image classification from the [in-class Kaggle challenge](https://www.kaggle.com/c/cs-ioc5008-hw1) and [NCTU Computer Vision HW](https://github.com/yan-roo/NCTU_Computer_Vision/blob/master/HW5/CV2020HW5.pdf).<br> 
The dataset is a little different:
* Kaggle challenge: 3859 grey images with 13 categories (train:2819, test:1040)
* CV HW: 1650 grey images with 15 categories (train:1500, test:150)

## Previous work
1. VGG16 (imagenet pretrain) + 2*FC layers & Dropout
2. ResNet50 (imagenet pretrain) on Keras 2.2.4 [Broken BatchNorm Freeze](http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/#comment-22015)
3. Image Size: **224**, VGG16 preprocess_input + horizontal_flip (on-the-fly data augmentation)
4. Train on spilt training set(loss some of training data)
5. Ensemble prediction on Kaggle 0.899 accuracy


## New method
1. ResNet50 (imagenet pretrain) on TF2.2 [classification_models](https://github.com/qubvel/classification_models)
2. CosineAnnealingScheduler
3. Image Size: **256**,  + horizontal_flip + brightness + zoom + rotation (on-the-fly data augmentation)
4. Train on whole training set
5. Single model prediction on CV HW 0.98 accuracy


## Experiment
### EfficientNet
|Model| Batch_size| Accuracy| Extra| 
| -------------- | :-: |:----:|:---:|
| EfficientNetB0 | 64  | 0.92 ||
| EfficientNetB0 | 64  | 0.906 |noisy-student pretrain|
| EfficientNetB1 | 64  |0.926 ||
| EfficientNetB1 | 64  | 0.906 |noisy-student pretrain|
| EfficientNetB4 | 16  | 0.92 ||
| EfficientNetB4 | 32  | 0.95 ||
| EfficientNetB4 | 32  | 0.89 |Freeze 1st Block(Conv+BN+Activation)|
| EfficientNetB4 | 32  | 0.9  |Freeze 1~2 Blocks(Conv+BN+Activation)|
| EfficientNetB5 | 16  | 0.926||
| EfficientNetB6 | 16  | 0.9  |Freeze 1st Block(Conv+BN+Activation)|
| EfficientNetB6 | 16  | 0.926|Freeze 1~2 Blocks(Conv+BN+Activation)|
| EfficientNetB6 | 16  | 0.94 |Freeze 1~3 Blocks(Conv+BN+Activation)|
| EfficientNetB6 | 16  | 0.85 |Freeze 1~4 Blocks(Conv+BN+Activation)|

### ResNet50
Freeze first 12 layers (0~47 layers in the implment)
|Model| Batch_size| Accuracy| Extra| 
| -------------- | :-: |:----:|:---:|
| ResNet50 | 64  | 0.953 |Generate New Data|
| ResNet50 | 64  | 0.966 |on-the-fly|
| ResNet50 | 64  | 0.946 |on-the-fly + constrast_pil|
| ResNet50 | 64  | 0.98  |on-the-fly + rotation 5|
| ResNet50 | 64  | 0.96  |on-the-fly + rotation 7|
| ResNet50 | 64  | 0.953 |on-the-fly + rotation 10|

## Conclusion
* Use ResNet50 with imagenet pretrain and freeze first 12 layers
* Large batch size might be helpful
* Use on-the-fly (random) instead of generate new data on data augmentation
* Use Brightness, Zoom and Rotation instead of Equalize and RandomResizedCropped
* Use TF2 if you want to freeze BN layers
* Sparse labels might help on accuracy (Dense without softmax, class_mode='sparse', loss=SparseCategoricalCrossentropy)

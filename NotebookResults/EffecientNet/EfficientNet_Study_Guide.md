# EfficientNet-B0 — Driver Drowsiness Detection: Study Guide

> **Goal:** Classify images into 6 drowsiness-related states using a pretrained CNN (EfficientNet-B0) via Transfer Learning.
> **Classes:** `yawn`, `no_yawn`, `Closed` (eyes), `Open` (eyes), `front` (head), `down` (head)

---

## Table of Contents
1. [Imports & Setup](#1-imports--setup)
2. [Data Loading](#2-data-loading)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Model Architecture](#4-model-architecture)
5. [Training — Phase 1 (Head Only)](#5-training--phase-1-head-only)
6. [Training — Phase 2 (Fine-Tuning)](#6-training--phase-2-fine-tuning)
7. [Evaluation & Analysis](#7-evaluation--analysis)
8. [Inference](#8-inference)

---

## 1. Imports & Setup

```python
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
```

Standard libraries. `cv2` is OpenCV — used for reading images and face detection.

```python
IMG_SIZE = 224
```

**Why 224?** EfficientNet-B0 was originally designed to take 224×224 pixel images as input (same as many ImageNet models). Every image in the dataset will be resized to this exact size.

---

## 2. Data Loading

The dataset has 3 separate sources that are loaded differently based on what they contain.

---

### 2a. Yawn / No-Yawn — Face Crop via Haar Cascade

```python
def face_for_yawn(direc="...", face_cas_path="..."):
    yaw_no = []
    categories = ["yawn", "no_yawn"]
    for category in categories:
        path_link = os.path.join(direc, category)
        class_num1 = categories.index(category)   # yawn=0, no_yawn=1
        for image in os.listdir(path_link):
            image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
            face_cascade = cv2.CascadeClassifier(face_cas_path)
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            for (x, y, w, h) in faces:
                img = cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_color = img[y:y+h, x:x+w]         # crop just the face
                resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
                yaw_no.append([resized_array, class_num1])
    return yaw_no
```

**Key concepts:**

| Term | Meaning |
|---|---|
| **Haar Cascade** | A classical (non-deep-learning) face detector. It slides a window across the image and uses pre-trained patterns ("features") to decide if a region is a face. Fast but less accurate than modern methods. |
| `cv2.CascadeClassifier` | Loads the Haar Cascade model from an XML file. |
| `detectMultiScale(img, 1.3, 5)` | Detects faces at multiple scales. `1.3` = scale factor (how much the window grows each step), `5` = minimum number of neighbors a region needs to be confirmed as a face. |
| `(x, y, w, h)` | Bounding box: top-left corner `(x,y)`, width `w`, height `h`. |
| `roi_color = img[y:y+h, x:x+w]` | **ROI (Region of Interest)** — crops just the face out of the full image. |
| `cv2.resize(roi_color, (224, 224))` | Scales the cropped face to the model's expected input size. |
| `class_num1 = categories.index(category)` | Assigns label: yawn → 0, no_yawn → 1. |

**Why crop the face for yawn?** The mouth region carries the yawn signal. Cropping removes irrelevant background and makes classification easier.

---

### 2b. Closed / Open Eyes

```python
def get_data(dir_path="...", ...):
    labels = ['Closed', 'Open']
    data = []
    for label in labels:
        path = os.path.join(dir_path, label)
        class_num = labels.index(label) + 2    # Closed=2, Open=3
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append([resized_array, class_num])
    return data
```

Simpler than yawn — no face cropping needed since these images are already eye-region crops. Labels are offset by +2 so they don't clash with yawn/no_yawn (which used 0 and 1).

---

### 2c. Head Pose — Front / Down

```python
def get_head_pose_data(dir_path="..."):
    categories = ["front", "down"]
    head_data = []
    for category in categories:
        class_num = categories.index(category) + 4   # front=4, down=5
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            if img_array is None:
                continue
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            if resized_array.shape != (IMG_SIZE, IMG_SIZE, 3):
                continue
            head_data.append([resized_array, class_num])
    return head_data
```

Same pattern. Labels offset to 4 and 5. Extra safety checks (`is None`, shape check) because this dataset had some corrupt/unreadable files.

---

### 2d. Combining All Data

```python
def append_data():
    yaw_no = face_for_yawn()
    data = get_data()
    head_data = get_head_pose_data()
    yaw_no.extend(data)
    yaw_no.extend(head_data)
    features = np.array([item[0] for item in yaw_no])
    labels   = np.array([item[1] for item in yaw_no])
    return list(zip(features, labels))

new_data = append_data()
```

All three datasets are merged into one list. Each element is a `(image_array, label_number)` pair. The final label mapping is:

```
0 = yawn    1 = no_yawn    2 = Closed    3 = Open    4 = front    5 = down
```

---

## 3. Data Preprocessing

### 3a. Separate Features and Labels

```python
X = []
y = []
for feature, label in new_data:
    X.append(feature)
    y.append(label)
```

Just unpacks the zipped list into two separate lists: `X` = images, `y` = labels.

---

### 3b. Reshape X

```python
X = np.array(X)
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
```

| Term | Meaning |
|---|---|
| `np.array(X)` | Converts the Python list of images into a NumPy array for efficient math. |
| `.reshape(-1, 224, 224, 3)` | Forces the shape to be `(N, 224, 224, 3)`. The `-1` means "figure out N automatically". The `3` = RGB channels. |

**Shape convention for CNNs:** `(batch_size, height, width, channels)` — Keras expects this exact format.

---

### 3c. One-Hot Encoding with LabelBinarizer

```python
from sklearn.preprocessing import LabelBinarizer
label_bin = LabelBinarizer()
y = label_bin.fit_transform(y)
```

**One-Hot Encoding:** Converts a single integer label into a binary vector.

```
label 2 (Closed) → [0, 0, 1, 0, 0, 0]
label 4 (front)  → [0, 0, 0, 0, 1, 0]
```

**Why?** The output layer uses `softmax`, which outputs 6 probabilities. The loss function (`categorical_crossentropy`) compares those 6 probabilities against a one-hot vector — you can't compare against a single integer directly.

---

### 3d. Train/Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.30)
```

| Term | Meaning |
|---|---|
| `test_size=0.30` | 30% of data held back for evaluation, 70% for training. |
| `random_state=42` | Fixed random seed so the split is reproducible every run. |

Result: `len(X_test) = 1847` samples for testing.

---

## 4. Model Architecture

### 4a. Data Augmentation

```python
train_generator = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True, rotation_range=30)
test_generator  = ImageDataGenerator()

train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
test_generator  = test_generator.flow(np.array(X_test),  y_test,  shuffle=False)
```

**Data Augmentation:** Artificially increases training variety by randomly transforming images on-the-fly during training. The original images are not changed — new transformed versions are generated each epoch.

| Parameter | Effect |
|---|---|
| `zoom_range=0.2` | Randomly zooms in/out by up to 20%. |
| `horizontal_flip=True` | Randomly flips images left-right. |
| `rotation_range=30` | Randomly rotates images up to ±30°. |

The test generator has no augmentation — test data should be clean and unmodified.

> **Note:** No `rescale=1./255` here. EfficientNetB0 includes its own internal normalization layer, so you pass raw pixel values (0–255) directly.

---

### 4b. Transfer Learning Concept

Instead of training a CNN from scratch, we use **EfficientNetB0** that was already trained on **ImageNet** (1.2 million images, 1000 classes). This pretrained model already knows how to detect low-level features (edges, textures, shapes). We keep that knowledge and only teach it our 6-class problem.

**Transfer Learning pipeline:**
```
[Pretrained EfficientNetB0 base] → [Our custom classification head]
      (frozen in Phase 1)               (trained first)
      (partially unfrozen in Phase 2)   (continues training)
```

---

### 4c. Model Definition

```python
base_model = EfficientNetB0(
    weights='imagenet',       # load pretrained weights
    include_top=False,        # remove the original 1000-class output layer
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # freeze all base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(6, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
```

| Term | Meaning |
|---|---|
| `weights='imagenet'` | Downloads weights pretrained on ImageNet. |
| `include_top=False` | Strips off EfficientNet's original final classification layer so we can attach our own. |
| `base_model.trainable = False` | **Freezing** — prevents the base model's weights from updating during Phase 1. Only our new layers train. |
| **GlobalAveragePooling2D** | Takes the feature maps output by the CNN (shape e.g. `7×7×1280`) and averages each channel into a single number. Result: a 1D vector of 1280 values. Replaces `Flatten` — more robust, far fewer parameters. |
| `Dense(256, activation='relu')` | A fully connected layer with 256 neurons. **ReLU** (Rectified Linear Unit) = `max(0, x)` — introduces non-linearity so the model can learn complex patterns, not just linear ones. |
| **Dropout(0.5)** | During training, randomly sets 50% of neuron outputs to 0 each forward pass. Forces the network to not rely on any single neuron — reduces **overfitting**. Disabled automatically during inference. |
| `Dense(6, activation='softmax')` | Output layer with 6 neurons (one per class). **Softmax** converts raw scores into probabilities that sum to 1.0. The highest probability = predicted class. |

```python
model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=Adam(learning_rate=1e-3)
)
```

| Term | Meaning |
|---|---|
| **Categorical Crossentropy** | Loss function for multi-class classification with one-hot labels. Measures how far the model's predicted probability distribution is from the true label. Lower = better. |
| **Adam optimizer** | Adaptive gradient descent. Adjusts the learning rate per parameter automatically. One of the most popular optimizers. |
| `learning_rate=1e-3` | How big a step the optimizer takes when updating weights. `1e-3 = 0.001`. |

---

## 5. Training — Phase 1 (Head Only)

```python
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history1 = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    callbacks=[early_stop, reduce_lr]
)
```

| Term | Meaning |
|---|---|
| **Epoch** | One full pass through the entire training dataset. |
| **Validation data** | A separate set the model never trains on — used to measure real-world performance after each epoch. |
| **EarlyStopping** | Stops training automatically if `val_loss` doesn't improve for `patience=5` consecutive epochs. `restore_best_weights=True` reloads the best checkpoint found. Prevents wasted time and overfitting. |
| **ReduceLROnPlateau** | If `val_loss` doesn't improve for `patience=3` epochs, multiplies the learning rate by `factor=0.5`. Helps escape plateaus. `min_lr=1e-6` is the floor. |

**Phase 1 results:** The model reaches ~98.9% training accuracy, ~98.4% validation accuracy in 13 epochs (early stopping triggered). Starting this high makes sense — the base model already knows useful features, so just training the head is very effective quickly.

---

## 6. Training — Phase 2 (Fine-Tuning)

```python
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False   # keep all layers frozen EXCEPT the last 30

model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=Adam(learning_rate=1e-5)   # much smaller lr than Phase 1
)

history2 = model.fit(train_generator, epochs=40, ...)
```

**Why fine-tune?** After training the head, the base model's deep layers still encode ImageNet patterns. Fine-tuning allows the last few layers to adapt to your specific dataset (faces, eyes, head poses) — squeezing out more accuracy.

**Why only the last 30 layers?** Early layers detect generic features (edges, colors) that are useful for any vision task — no need to change them. The last layers detect higher-level, task-specific features — those benefit from updating.

**Why `learning_rate=1e-5`?** The pretrained weights are already good. A large learning rate would destroy them. This tiny rate (`0.00001`) makes very small, careful adjustments.

**Phase 2 results:** Val accuracy improves further to ~98.9%. The model trained for the full 40 epochs as early stopping wasn't triggered (val_loss kept slowly improving).

---

## 7. Evaluation & Analysis

### 7a. Training History Plot

```python
acc     = history1.history['accuracy']     + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
# ... (same for loss)
plt.axvline(x=len(history1.history['accuracy']), color='gray', linestyle='--', label='Fine-tune start')
```

Concatenates both training phases into one continuous curve. The vertical dashed line marks where Phase 2 began. This lets you visually see the effect of fine-tuning on top of head-only training.

---

### 7b. Classification Report

```python
prediction = np.argmax(model.predict(X_test), axis=1)

print(classification_report(np.argmax(y_test, axis=1), prediction, target_names=labels_new))
```

`np.argmax(..., axis=1)` — converts probability vectors back to class integers (picks the index of the highest value).

**Report output:**

```
              precision    recall  f1-score   support
        yawn       0.97      0.86      0.91        84
     no_yawn       0.87      0.97      0.92        76
      Closed       0.99      1.00      0.99       223
        Open       1.00      0.99      1.00       228
       front       1.00      0.99      0.99       594
        down       0.99      1.00      0.99       642

    accuracy                           0.99      1847
```

| Metric | Meaning |
|---|---|
| **Precision** | Of everything predicted as class X, what fraction was actually X? (Measures false positives) |
| **Recall** | Of all actual class X samples, what fraction did we correctly find? (Measures false negatives) |
| **F1-Score** | Harmonic mean of precision and recall. Use when both matter equally. `2 * (P * R) / (P + R)` |
| **Support** | Number of actual samples of that class in the test set. |

`yawn` and `no_yawn` are the hardest (smallest support, most confusion with each other — both involve the mouth).

---

### 7c. Confusion Matrix

```python
cm = confusion_matrix(true_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ...)
```

A grid where row = true label, column = predicted label. Diagonal = correct predictions. Off-diagonal = mistakes. Makes it easy to see *which* classes confuse the model.

---

### 7d. ROC Curves & AUC

```python
for i, label in enumerate(labels_new):
    fpr, tpr, _ = roc_curve(y_test[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC={roc_auc:.3f})')
```

**One-vs-Rest strategy:** For each class, treats it as a binary problem (this class vs. all others).

| Term | Meaning |
|---|---|
| **ROC Curve** | Plots True Positive Rate vs False Positive Rate at every possible classification threshold. |
| **AUC** | Area Under the Curve. Ranges 0–1. `1.0` = perfect classifier. `0.5` = random guessing. Higher is better. |
| `fpr, tpr` | False Positive Rate, True Positive Rate at each threshold. |

---

### 7e. Precision-Recall Curves

```python
for i, label in enumerate(labels_new):
    precision, recall, _ = precision_recall_curve(y_test[:, i], y_prob[:, i])
    ap = average_precision_score(y_test[:, i], y_prob[:, i])
```

| Term | Meaning |
|---|---|
| **PR Curve** | Plots Precision vs Recall at every threshold. More informative than ROC when classes are imbalanced. |
| **Average Precision (AP)** | Area under the PR curve. `1.0` = perfect. |

---

### 7f. Misclassification Gallery

```python
wrong_idx = np.where(predictions != true_labels)[0]
```

Finds indices where prediction ≠ true label. Then displays those images with both the true and predicted label so you can visually inspect what the model gets wrong.

---

### 7g. Grad-CAM

```python
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        class_score = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()
```

**Grad-CAM (Gradient-weighted Class Activation Mapping):** A technique to *visualize what part of the image the model focused on* when making a prediction. Produces a heatmap overlaid on the original image.

| Step | What's happening |
|---|---|
| Build `grad_model` | A model that outputs both the last Conv layer's feature maps AND the final prediction simultaneously. |
| `GradientTape` | Records operations so we can compute gradients (derivatives) later. |
| `tape.gradient(class_score, conv_outputs)` | Computes how much each feature map pixel affected the predicted class score. |
| `tf.reduce_mean(grads, axis=(0,1,2))` | Averages gradients across spatial dimensions → one importance weight per feature map channel. |
| `conv_outputs @ pooled_grads` | Weighted sum of feature maps. High values = regions the model found important. |
| `tf.maximum(heatmap, 0)` | ReLU — keeps only positive influences (regions that pushed *toward* this class). |

The overlay (`overlay_gradcam`) blends the colored heatmap onto the original image so you can see "the model looked at the mouth region when predicting yawn."

---

### 7h. t-SNE of Feature Embeddings

```python
feature_extractor = tf.keras.models.Model(
    inputs=model.input,
    outputs=model.get_layer(gap_layer_name).output   # 1280-dimensional vectors
)
embeddings = feature_extractor.predict(X_test[idx_subset])

tsne   = TSNE(n_components=2, random_state=42, perplexity=40)
emb_2d = tsne.fit_transform(embeddings)
```

**Feature Embeddings:** The 1280-dimensional vector at the GlobalAveragePooling2D layer is a compressed representation of the image — a "fingerprint" learned by the CNN.

**t-SNE (t-distributed Stochastic Neighbor Embedding):** A dimensionality reduction algorithm that projects high-dimensional data (1280D) down to 2D for visualization. It tries to preserve the neighborhood structure: similar points in 1280D appear close in 2D.

If the 2D scatter plot shows tight, well-separated clusters per class, it means the model has learned a very good internal representation. If clusters overlap, the model struggles to distinguish those classes.

| Term | Meaning |
|---|---|
| `n_components=2` | Project down to 2 dimensions (for 2D scatter plot). |
| `perplexity=40` | Controls how many neighbors t-SNE considers. Roughly: larger dataset → higher perplexity. |

---

## 8. Inference

```python
def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    resized   = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

model = tf.keras.models.load_model('drowsiness_efficientnet_b0.h5')

prediction = model.predict([prepare('path/to/image.jpg')])
np.argmax(prediction)   # returns 0–5
```

To use the trained model on a new image:
1. Read the image with OpenCV.
2. Resize to 224×224.
3. Add a batch dimension with `.reshape(-1, ...)` — models always expect a batch, even of size 1.
4. Call `.predict()` → returns a probability vector of shape `(1, 6)`.
5. `np.argmax()` picks the class with the highest probability.

```
0=yawn  1=no_yawn  2=Closed  3=Open  4=front  5=down
```

---

## Summary: Full Pipeline

```
Raw Images
    │
    ▼
Face/Eye/Head Detection (Haar Cascade)
    │
    ▼
Resize to 224×224  ──► Assign integer labels (0–5)
    │
    ▼
One-Hot Encode labels  ──► Train/Test Split (70/30)
    │
    ▼
Data Augmentation (zoom, flip, rotate)
    │
    ▼
Phase 1: EfficientNetB0 (frozen) + New Head  [lr=0.001]
    │
    ▼
Phase 2: Unfreeze last 30 layers, Fine-tune  [lr=0.00001]
    │
    ▼
Evaluate: Classification Report, Confusion Matrix,
          ROC/PR Curves, Grad-CAM, t-SNE
    │
    ▼
Final Accuracy: ~99%
```

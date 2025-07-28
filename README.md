#  Food-101 Image Classification with TensorFlow


This project involves building a deep learning model to classify images from the **Food-101** dataset into 101 different food categories. The model uses a custom Convolutional Neural Network (CNN) built with TensorFlow and Keras. The dataset is preprocessed, visualized, and augmented through a complete data pipeline, and the model is trained and evaluated using best practices like batching, caching, and early stopping.

---

##  **Overview**

- Download and extract the Food-101 dataset.
- Visualize samples from each food category.
- Prepare the dataset by splitting into training and testing folders.
- Preprocess and batch the images using TensorFlow `tf.data` API.
- Build and train a custom CNN model.
- Monitor training with TensorBoard and apply callbacks like early stopping and model checkpointing.

---

##  **Tools & Libraries**

- **TensorFlow / Keras**
- **TensorFlow Datasets**
- **NumPy**
- **Matplotlib**
- **Collections & OS utilities**
- **Shutil (for file operations)**

---

##  **Dataset**

- **Name**: [Food-101](https://data.vision.ee.ethz.ch/cvl/food-101/)
- **Description**: A dataset of 101 food categories with 1,000 images each.
- **Structure**:
  - `images/`: Folder with all food images grouped by class.
  - `meta/train.txt`: List of training image paths.
  - `meta/test.txt`: List of testing image paths.
  - `meta/classes.txt`: List of all 101 food categories.

---

##  **Pipeline**

###  1. **Dataset Setup**
- Check for dataset existence.
- Download `food-101.tar.gz` if not present.
- Extract dataset using `tar`.

###  2. **Visualization**
- Display one random image per class (up to 101 categories).
- Use `matplotlib` for plotting in grid layout.

###  3. **Dataset Preparation**
- Read image paths from `train.txt` and `test.txt`.
- Copy images into separate folders: `train/` and `test/`.

###  4. **Preprocessing**
- Resize all images to `224x224`.
- Normalize pixel values to the range `[0, 1]`.
- Convert dataset into `tf.data.Dataset` objects.

###  5. **Data Pipeline Optimization**
- Use `shuffle()` to randomize batches.
- Apply `batch()` to process data in batches.
- Use `prefetch()` to prepare the next batch while the current one is training.
- (Optional) Use `cache()` to speed up subsequent epochs.

###  6. **Visualization of Preprocessed Batches**
- Display a batch of 9 preprocessed images with labels to verify correctness.

###  7. **Model Architecture**
- Custom CNN with multiple Conv2D and MaxPooling2D layers.
- Flatten and Dense layers at the end.
- Final Dense layer with 101 units (softmax) for classification.

###  8. **Training**
- Compile the model with:
  - `Adam` optimizer
  - `Sparse Categorical Crossentropy` loss
  - `Accuracy` metric
- Use callbacks:
  - `TensorBoard` for training visualization
  - `EarlyStopping` to prevent overfitting
  - `ModelCheckpoint` to save the best model
- Train the model using the optimized data pipeline.

---

##  **Results**

- The model is trained for 10 epochs (can be tuned further).
- Training and validation performance is monitored via:
  - Accuracy
  - Loss
- TensorBoard logs are generated for real-time monitoring.

---

##  **Conclusion**

- The project demonstrates how to build an end-to-end image classification pipeline using TensorFlow.
- Data loading, preprocessing, batching, and caching ensure efficient training.
- The final model can classify food images into 101 categories with high accuracy.

---
-

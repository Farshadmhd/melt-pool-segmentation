import os
import numpy as np
import tensorflow as tf
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_float32
from sklearn.metrics import precision_score, recall_score, jaccard_score

# Define the custom metrics
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def iou(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def precision(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    true_positives = tf.keras.backend.sum(y_true_f * y_pred_f)
    predicted_positives = tf.keras.backend.sum(y_pred_f)
    return (true_positives + smooth) / (predicted_positives + smooth)

def recall(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    true_positives = tf.keras.backend.sum(y_true_f * y_pred_f)
    possible_positives = tf.keras.backend.sum(y_true_f)
    return (true_positives + smooth) / (possible_positives + smooth)

# Load the model with custom objects
def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={
        'dice_coefficient': dice_coefficient,
        'iou': iou,
        'precision': precision,
        'recall': recall
    })

# Preprocess image
def preprocess_image(image_path):
    image = imread(image_path, as_gray=True)
    image = img_as_float32(image)
    original_shape = image.shape[:2]  # Capture original dimensions
    image = resize(image, (768, 1200), mode='constant', preserve_range=True)
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image / np.max(image), original_shape

# Post-process the predictions
def post_process_predictions(predictions, original_shape, threshold=0.5):
    predictions = (predictions > threshold).astype(np.uint8)
    predictions = predictions.squeeze()  # Remove single-dimensional entries
    return resize(predictions, original_shape, mode='constant', preserve_range=True).astype(np.uint8)

# Overlay the mask on the original image with adjustable opacity
def overlay_mask_on_image(original_image, mask, alpha=0.5):
    # Ensure the original image is in the range [0, 1]
    original_image = original_image / np.max(original_image)
    
    overlay = np.dstack((original_image, original_image, original_image))  # Convert grayscale to RGB
    red_mask = np.dstack((mask, np.zeros_like(mask), np.zeros_like(mask)))  # Red mask
    
    # Blend images
    blended = overlay * (1 - alpha) + red_mask * alpha
    return blended

# Calculate metrics for a single image
def calculate_metrics(y_true, y_pred):
    # Flatten the arrays
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    # Calculate metrics
    precision = precision_score(y_true_f, y_pred_f, average='binary')
    recall = recall_score(y_true_f, y_pred_f, average='binary')
    iou = jaccard_score(y_true_f, y_pred_f, average='binary')
    
    return precision, recall, iou

# Define paths
model_path = '/work/cf_farshad/meltpool_segmentation_model33.h5'  # Updated to match your training script
source_folder = '/work/cf_farshad/Label/useful'
output_folder = '/work/cf_farshad/Label/Prediction_33'

# Load the trained model
model = load_model(model_path)

# Ensure output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize lists to store metrics
precisions = []
recalls = []
ious = []

# Process each image in the source directory
for filename in os.listdir(source_folder):
    if filename.endswith('.png'):  # Check for PNG files, change this as necessary
        image_path = os.path.join(source_folder, filename)
        output_path = os.path.join(output_folder, 'overlay_' + filename)

        # Preprocess the image
        image, original_shape = preprocess_image(image_path)

        # Predict the mask
        predictions = model.predict(image)

        # Post-process the predictions
        mask = post_process_predictions(predictions, original_shape)

        # Load the original image for overlay
        original_image = imread(image_path, as_gray=True)
        original_image = img_as_float32(original_image)

        # Overlay the mask on the original image with specified opacity
        alpha = 0.5  # Adjust opacity here (0.0 to 1.0)
        overlay = overlay_mask_on_image(original_image, mask, alpha)

        # Save the overlay image
        overlay = (overlay * 255).astype(np.uint8)  # Ensure the overlay is in the correct format
        imsave(output_path, overlay, check_contrast=False)

        # Calculate metrics
        precision, recall, iou = calculate_metrics(original_image > 0, mask > 0)  # Assuming binary ground truth
        precisions.append(precision)
        recalls.append(recall)
        ious.append(iou)

        # Check if file was saved
        if os.path.exists(output_path):
            print(f"Processed and saved overlay for {filename}")
        else:
            print(f"Failed to save overlay for {filename}")

# Save the metrics to a file
metrics_path = os.path.join(output_folder, 'metrics.txt')
with open(metrics_path, 'w') as f:
    for i, filename in enumerate(os.listdir(source_folder)):
        if filename.endswith('.png'):
            f.write(f"{filename}: Precision={precisions[i]}, Recall={recalls[i]}, IoU={ious[i]}\n")

# Calculate and print average metrics
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_iou = np.mean(ious)

print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Average IoU: {avg_iou}")

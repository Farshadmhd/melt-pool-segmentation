import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Activation, Add, Multiply
from tensorflow.keras.optimizers import Adam
from skimage.io import imread
from skimage.transform import resize
from skimage import img_as_float32
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train Attention U-Net for melt-pool segmentation.')
parser.add_argument('--train-path', type=str, default='/work/cf_farshad/Label/Train_orginal', help='Path to training images.')
parser.add_argument('--label-path', type=str, default='/work/cf_farshad/Label/Mask', help='Path to corresponding masks.')
parser.add_argument('--model-save-path', type=str, default='/work/cf_farshad/meltpool_segmentation_model34.h5', help='Path to save the trained model.')
args = parser.parse_args()

# Set environment variables to allow TensorFlow to automatically tune the best algorithm
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'

# Check available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Dice Coefficient Metric
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Intersection over Union (IoU)
def iou(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# Precision
def precision(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positives = K.sum(y_true_f * y_pred_f)
    predicted_positives = K.sum(y_pred_f)
    return (true_positives + smooth) / (predicted_positives + smooth)

# Recall
def recall(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positives = K.sum(y_true_f * y_pred_f)
    possible_positives = K.sum(y_true_f)
    return (true_positives + smooth) / (possible_positives + smooth)

# Attention block definition
def attention_block(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, (1, 1), strides=(2, 2), padding='same')(x)
    phi_g = Conv2D(inter_channel, (1, 1), padding='same')(g)
    add_xg = Add()([theta_x, phi_g])
    act_xg = Activation('relu')(add_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    upsample_psi = UpSampling2D(size=(2, 2))(sigmoid_xg)
    y = Multiply()([upsample_psi, x])
    result = Conv2D(x.shape[-1], (1, 1), padding='same')(y)
    result_bn = Activation('relu')(result)
    return result_bn

# Attention U-Net architecture
def attention_unet(input_size=(768, 1200, 1)):
    inputs = Input(input_size)
    s = inputs

    # Encoding path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Decoding path
    g4 = attention_block(c4, c5, 128)
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, g4], axis=3)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    g3 = attention_block(c3, c6, 64)
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, g3], axis=3)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    g2 = attention_block(c2, c7, 32)
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, g2], axis=3)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    g1 = attention_block(c1, c8, 16)
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, g1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Configuration parameters
TARGET_WIDTH = 1200  # Resize target width
TARGET_HEIGHT = 768  # Resize target height
IMG_CHANNELS = 1  # Set to 1 for grayscale images
TRAIN_PATH = args.train_path
LABEL_PATH = args.label_path

def load_data(train_path, label_path, target_height=768, target_width=1200):
    X = []
    Y = []
    for filename in os.listdir(train_path):
        if filename.endswith('.png'):
            img_path = os.path.join(train_path, filename)
            label_file_path = os.path.join(label_path, filename)
            try:
                img = imread(img_path, as_gray=True)  # Read the image
                label = imread(label_file_path, as_gray=True)  # Read the corresponding label

                img = img_as_float32(img)
                label = img_as_float32(label)

                # Resize images to the target size
                img = resize(img, (target_height, target_width), mode='constant', preserve_range=True)
                label = resize(label, (target_height, target_width), mode='constant', preserve_range=True)

                # Add channel dimension if it's not present
                img = np.expand_dims(img, axis=-1)
                label = np.expand_dims(label, axis=-1)

                X.append(img)
                Y.append(label)
            except Exception as e:
                logging.error(f"Error processing file {filename}: {str(e)}")

    if not X:
        logging.warning("No valid images found.")
    else:
        X = np.array(X)
        Y = np.array(Y)
        # Normalize the data arrays
        max_X = np.max(X) if X.size > 0 else 1.0  # If X is empty, set max_X to 1.0 to avoid division by zero
        max_Y = np.max(Y) if Y.size > 0 else 1.0  # If Y is empty, set max_Y to 1.0 to avoid division by zero
        X = X / max_X
        Y = Y / max_Y

    return X, Y

# Initialize Attention U-Net model
input_size = (768, 1200, 1)  # Adjusting the channel as needed
with tf.device('/GPU:0'):  # Specify GPU
    model = attention_unet(input_size)

# Define the optimizer with the desired learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Compile the model with the defined optimizer and metrics
model.compile(optimizer=optimizer, loss='binary_crossentropy', 
              metrics=[dice_coefficient, iou, precision, recall])
model.summary()

# Load and preprocess data
X_train, Y_train = load_data(TRAIN_PATH, LABEL_PATH)

# Initialize ReduceLROnPlateau callback
reduce_lr = ReduceLROnPlateau(monitor='val_dice_coefficient', factor=0.5, patience=5, min_lr=1e-2, verbose=1)

# Train the model with a fixed number of epochs and learning rate scheduler
with tf.device('/GPU:0'):  # Specify GPU
    history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=10, epochs=100, 
                        callbacks=[reduce_lr])

# Save the model
model_save_path = args.model_save_path
model.save(model_save_path)

# Plot the metrics and save the plots
plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
plt.plot(history.history['dice_coefficient'])
plt.plot(history.history['val_dice_coefficient'])
plt.title('Model Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(2, 2, 2)
plt.plot(history.history['iou'])
plt.plot(history.history['val_iou'])
plt.title('Model IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(2, 2, 3)
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('Model Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(2, 2, 4)
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('Model Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend(['Train', 'Validation'], loc='upper left')

plot_save_path = os.path.splitext(model_save_path)[0] + '_metrics_plot.png'
plt.savefig(plot_save_path)

# Display the plot
plt.show()

# Check the number of epochs completed
completed_epochs = len(history.history['loss'])
print(f"Training completed after {completed_epochs} epochs.")

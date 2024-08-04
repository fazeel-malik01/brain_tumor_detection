import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define constants
image_size = 150
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
model_path = 'classifier/trained_models/model.h5'

def load_history():
    history_path = 'classifier/trained_models/history.npy'
    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True).item()
        return history
    else:
        print("No history file found.")
        return None

def plot_training_history(history):
    if history is not None:
        # Plot accuracy
        plt.figure()
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(os.path.join('Reports', 'training_accuracy.png'))
        print("Saved training_accuracy.png")

        # Plot loss
        plt.figure()
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(os.path.join('Reports', 'training_loss.png'))
        print("Saved training_loss.png")
    else:
        print("No history data to plot.")

def load_data():
    base_path = 'dataset/Testing'
    X_test = []
    Y_test = []

    for label in labels:
        folder_path = os.path.join(base_path, label)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (image_size, image_size))
                X_test.append(img)
                Y_test.append(label)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    Y_test = tf.keras.utils.to_categorical([labels.index(i) for i in Y_test])
    X_test = X_test / 255.0
    return X_test, Y_test

def generate_confusion_matrix():
    model = load_model(model_path)

    # Load the test data
    X_test, Y_test = load_data()

    # Predictions
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(Y_test, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Actual', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    plt.savefig(os.path.join('Reports', 'confusion_matrix.png'), bbox_inches='tight')
    print("Saved confusion_matrix.png")

# Load history and plot training history
history = load_history()
plot_training_history(history)

# Generate and save confusion matrix
generate_confusion_matrix()

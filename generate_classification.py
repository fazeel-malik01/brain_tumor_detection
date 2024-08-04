import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

# Define constants
image_size = 150
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
model_path = 'classifier/trained_models/model.h5'

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

def generate_report():
    # Load the model
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
    plt.savefig('confusion_matrix.png', bbox_inches='tight')

    # Classification Report
    class_report_dict = classification_report(y_true, y_pred_classes, target_names=labels, output_dict=True)
    class_report_df = pd.DataFrame(class_report_dict).transpose()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(class_report_df.iloc[:-1, :].T, annot=True, cmap='Blues', annot_kws={"size": 14}, fmt='.2f')
    plt.title('Classification Report', fontsize=16)
    plt.savefig(os.path.join('Reports', 'classification_report.png'), bbox_inches='tight')

    print("Confusion matrix and classification report saved as images.")

if __name__ == '__main__':
    generate_report()

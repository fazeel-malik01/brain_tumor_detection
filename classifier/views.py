from django.shortcuts import render
from django.conf import settings
from django.shortcuts import render
import os
import numpy as np
import cv2
import tensorflow as tf
from django.http import JsonResponse
from sklearn.model_selection import train_test_split
from django.views.decorators.csrf import csrf_exempt
from sklearn.utils import shuffle
from keras.models import Sequential, save_model, load_model
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications import VGG16
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# Define constants
image_size = 150
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def load_data():
    base_path = os.path.join(os.getcwd(), 'dataset')
    X_train = []
    Y_train = []

    for label in labels:
        for dataset_type in ['Training', 'Testing']:
            folder_path = os.path.join(base_path, dataset_type, label)
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"The directory {folder_path} does not exist.")
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if os.path.isfile(img_path):  # Ensure it's a file
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (image_size, image_size))
                    X_train.append(img)
                    Y_train.append(label)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    X_train, Y_train = shuffle(X_train, Y_train, random_state=101)
    return train_test_split(X_train, Y_train, test_size=0.1, random_state=101)

def preprocess_labels(y_train, y_test):
    y_train = tf.keras.utils.to_categorical([labels.index(i) for i in y_train])
    y_test = tf.keras.utils.to_categorical([labels.index(i) for i in y_test])
    return np.array(y_train), np.array(y_test)

def create_datagen():
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,  
        fill_mode='nearest'
    )

def create_transfer_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model

def train_model(request):
    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_data()
        y_train, y_test = preprocess_labels(y_train, y_test)
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        
        # Data augmentation
        datagen = create_datagen()
        
        # Early stopping and learning rate reduction
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7)
        
        # Create model
        model = create_transfer_model()
        
        # Train model
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=5,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr]
        )
        
        # Save the trained model 
        model_save_path = os.path.join('classifier', 'trained_models', 'model.h5')
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        save_model(model, model_save_path)
        
        # Save the training history
        history_save_path = os.path.join('classifier', 'trained_models', 'history.npy')
        np.save(history_save_path, history.history)
        
        # Evaluate model
        y_true = np.argmax(y_test, axis=1)
        y_pred_probs = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_true, y_pred_classes)
        cm = confusion_matrix(y_true, y_pred_classes)
        class_report = classification_report(y_true, y_pred_classes, output_dict=True)
        
        # Return the evaluation results
        return JsonResponse({
            "status": "Training and evaluation complete",
            "evaluation_results": {
                "accuracy": accuracy,
                "confusion_matrix": cm.tolist(),
                "classification_report": class_report
            }
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
def continue_training(request, additional_epochs=1):
    try:
        # Load the saved model
        model_path = os.path.join('classifier', 'trained_models', 'model.h5')
        model = load_model(model_path)
        
        # Load the training data
        X_train, X_test, y_train, y_test = load_data()
        y_train, y_test = preprocess_labels(y_train, y_test)
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        
        # Data augmentation
        datagen = create_datagen()
        
        # Load the previous history
        history_save_path = os.path.join('classifier', 'trained_models', 'history.npy')
        if os.path.exists(history_save_path):
            previous_history = np.load(history_save_path, allow_pickle=True).item()
        else:
            previous_history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
        
        # Continue training
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7)
        
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=additional_epochs,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr]
        )
        
        # Update and save the history
        for key in history.history.keys():
            previous_history[key].extend(history.history[key])
        np.save(history_save_path, previous_history)
        
        # Save the updated model
        save_model(model, model_path)
        
        # Evaluate model
        y_true = np.argmax(y_test, axis=1)
        y_pred_probs = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_true, y_pred_classes)
        cm = confusion_matrix(y_true, y_pred_classes)
        class_report = classification_report(y_true, y_pred_classes, output_dict=True)
        
        # Return the evaluation results
        return JsonResponse({
            "status": "Continued training and evaluation complete",
            "evaluation_results": {
                "accuracy": accuracy,
                "confusion_matrix": cm.tolist(),
                "classification_report": class_report
            }
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

model_path = 'classifier/trained_models/model.h5'
model = load_model(model_path)
@csrf_exempt
def predict(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Read the uploaded image file
            image_file = request.FILES['image']
            image = np.asarray(bytearray(image_file.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (150, 150))
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            # Make a prediction
            predictions = model.predict(image)
            predicted_class = np.argmax(predictions[0])
            result = labels[predicted_class]

            # Log the result
            print(f"Predicted result: {result}")

            return JsonResponse({"result": result})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request"}, status=400)

@csrf_exempt
def show_metrics(request):
    try:
        # Load the test dataset
        base_path = os.path.join(settings.BASE_DIR, 'dataset', 'testing')
        X_test = []
        Y_test = []

        for label in labels:
            folder_path = os.path.join(base_path, label)
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"The directory {folder_path} does not exist.")
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if os.path.isfile(img_path):  # Ensure it's a file
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (image_size, image_size))
                    X_test.append(img)
                    Y_test.append(label)

        X_test = np.array(X_test)
        Y_test = np.array(Y_test)

        # Preprocess the labels
        Y_test = tf.keras.utils.to_categorical([labels.index(i) for i in Y_test])

        # Normalize the data
        X_test = X_test / 255.0

        # Make predictions
        y_pred_probs = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(Y_test, axis=1)

        # Compute accuracy, precision, recall, and F1 score
        accuracy = accuracy_score(y_true, y_pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average='weighted', zero_division=0)

        # If F1 is undefined or NaN, set it to 0
        if np.isnan(f1):
            f1 = 0.0

        # Prepare the results
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        return JsonResponse({"status": "Metrics computed successfully", "metrics": metrics})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def index(request):
    return render(request, 'classifier/index.html')



    
import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation
import os  # Import os for file path operations
from PIL import Image  # Import PIL for image processing
import tensorflow as tf  # Import TensorFlow for deep learning
from tensorflow.keras.models import Sequential  # Import Sequential model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout  # Import layers
from tensorflow.keras.utils import to_categorical  # Import utility for one-hot encoding
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score  # Import metrics
from tqdm import tqdm  # Import tqdm for progress bars
import csv  # Import csv for writing results to CSV files

# Base directory for images
base_image_dir = "C:/Users/nikhil/Desktop/CS412/Fiji Data/Trial"

# Load CSV files
train_csv_path = os.path.join(base_image_dir, 'Train.csv')
validation_csv_path = os.path.join(base_image_dir, 'Validation.csv')  # Validation CSV file
test_csv_paths = [
    os.path.join(base_image_dir, 'Day.csv'),
    os.path.join(base_image_dir, 'Night.csv'),
    os.path.join(base_image_dir, 'M5Day.csv'),
    os.path.join(base_image_dir, 'M5Night.csv'),
    os.path.join(base_image_dir, 'M10Day.csv'),
    os.path.join(base_image_dir, 'M10Night.csv'),
    os.path.join(base_image_dir, 'M15Day.csv'),
    os.path.join(base_image_dir, 'M15Night.csv'),
    os.path.join(base_image_dir, 'M20Day.csv'),
    os.path.join(base_image_dir, 'M20Night.csv'),
    os.path.join(base_image_dir, 'BlurDay-High.csv'),
    os.path.join(base_image_dir, 'BlurDay-Medium.csv'),
    os.path.join(base_image_dir, 'BlurNight-High.csv'),
    os.path.join(base_image_dir, 'BlurNight-Medium.csv'),
    os.path.join(base_image_dir, 'FogDay - High.csv'),
    os.path.join(base_image_dir, 'FogDay - Low.csv'),
    os.path.join(base_image_dir, 'FogDay-Medium.csv'),
    os.path.join(base_image_dir, 'FogNight-High.csv'),
    os.path.join(base_image_dir, 'FogNight-Low.csv'),
    os.path.join(base_image_dir, 'FogNight-Medium.csv')
]

train_df = pd.read_csv(train_csv_path)  # Load training data
validation_df = pd.read_csv(validation_csv_path)  # Load validation data

# Function to load images from a dataframe
def load_images(dataframe, base_dir):
    data = []  # Initialize list for image data
    labels = []  # Initialize list for labels
    error_count = 0  # Initialize error counter
    for idx, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc="Loading images"):
        relative_path = row['Path'].replace('\\', '/')  # Normalize file path
        img_path = os.path.join(base_dir, relative_path).replace('\\', '/')
        
        if not os.path.exists(img_path):  # Check if file exists
            print(f"File not found: {img_path}")
            continue  # Skip if not found
        try:
            image = Image.open(img_path).convert('RGB')  # Open and convert image to RGB
            image = image.resize((30, 30))  # Resize image
            data.append(np.array(image))  # Add image data to list
            labels.append(row['ClassId'] - 1)  # Add label to list (adjusted to start from 0)
        except Exception as e:  # Handle exceptions
            print(f"Error loading image {img_path}: {e}")
            error_count += 1
            continue
    print(f"Total images loaded: {len(data)}")  # Print total images loaded
    print(f"Total loading errors: {error_count}")  # Print total errors
    return np.array(data), np.array(labels)  # Return data and labels as numpy arrays

# Function to train, evaluate, and save results
def train_and_evaluate(run_number, metrics_writer):
    # Load training data
    X_train, y_train = load_images(train_df, base_image_dir)
    print(f"Loaded training data shape: {X_train.shape}")
    print(f"Number of training images loaded: {X_train.shape[0]}")

    # Load validation data
    X_val, y_val = load_images(validation_df, base_image_dir)
    print(f"Loaded validation data shape: {X_val.shape}")
    print(f"Number of validation images loaded: {X_val.shape[0]}")

    # Verify class labels
    unique_labels = np.unique(y_train)
    print(f"Unique class labels in training data: {unique_labels}")

    # Ensure all labels are within the range [0, 5]
    assert np.all(unique_labels < 6), "Error: Found class labels outside the range [0, 5]"

    # Converting the labels into one hot encoding
    y_train = to_categorical(y_train, 6)  # Updated to 6 classes
    y_val = to_categorical(y_val, 6)  # Updated to 6 classes for validation

    with tf.device('/GPU:0'):  # Use GPU for training
        # Model Building
        model = Sequential([
            Conv2D(32, (5, 5), activation='relu', input_shape=X_train.shape[1:]),  # First convolutional layer
            Conv2D(32, (5, 5), activation='relu'),  # Second convolutional layer
            MaxPool2D(pool_size=(2, 2)),  # First max pooling layer
            Dropout(0.25),  # First dropout layer
            Conv2D(64, (3, 3), activation='relu'),  # Third convolutional layer
            Conv2D(64, (3, 3), activation='relu'),  # Fourth convolutional layer
            MaxPool2D(pool_size=(2, 2)),  # Second max pooling layer
            Dropout(0.25),  # Second dropout layer
            Flatten(),  # Flatten layer
            Dense(256, activation='relu'),  # Dense layer with 256 units
            Dropout(0.5),  # Third dropout layer
            Dense(6, activation='softmax')  # Output layer with 6 units (classes)
        ])

        # Compilation of the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Displaying the model summary
        model.summary()

        # Training the model
        epochs = 15
        history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_val, y_val))

    # Save training and validation accuracy and loss
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    # Function to load and evaluate a test dataset and plot ROC curve
    def evaluate_test_dataset(test_csv_path, model, base_dir, run_number, metrics_writer):
        test_df = pd.read_csv(test_csv_path)  # Load test data
        X_test, y_test = load_images(test_df, base_dir)  # Load images
        y_test_cat = to_categorical(y_test, 6)  # Updated to 6 classes

        # Predict probabilities for ROC curve
        y_pred_prob = model.predict(X_test)

        # Predict classes
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Evaluate the model
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=2)
        print(f"Test accuracy for {os.path.basename(test_csv_path)}: {test_acc}")

        # Compute ROC curve and AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(6):  # For each class
            fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for each class
        plt.figure()
        for i in range(6):  # For each class
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {os.path.basename(test_csv_path)} - Run {run_number}')
        plt.legend(loc='lower right')
        plt.savefig(f'roc_curve_{os.path.basename(test_csv_path)}_run_{run_number}.png')
        plt.close()

        # Calculate additional metrics
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = recall  # In multi-class classification, sensitivity is equivalent to recall

        print(f"Metrics for {os.path.basename(test_csv_path)} - Run {run_number}:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")

        # Save metrics to the CSV file
        metrics_writer.writerow([run_number, os.path.basename(test_csv_path), train_acc, val_acc, train_loss, val_loss,
                                 precision, recall, f1, accuracy, sensitivity, test_loss, test_acc])

    # Evaluate the model on each test dataset
    for test_csv_path in test_csv_paths:
        evaluate_test_dataset(test_csv_path, model, base_image_dir, run_number, metrics_writer)

    # Save the model
    model.save(f'traffic_classifier_run_{run_number}.h5')

# Open CSV file for writing metrics
with open('model_metrics.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Run', 'Test_Set', 'Train_Accuracy', 'Validation_Accuracy', 'Train_Loss', 'Validation_Loss',
                     'Precision', 'Recall', 'F1_Score', 'Accuracy', 'Sensitivity', 'Test_Loss', 'Test_Accuracy'])

    # Run the process 30 times
    for run in range(1, 2):  # Run only once for demonstration
        print(f"Starting run {run}...")
        train_and_evaluate(run, writer)
        print(f"Completed run {run}.\n")

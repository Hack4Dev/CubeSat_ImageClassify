import os
import numpy as np
from PIL import Image

import time
import sys
import pickle
import psutil
import threading
import gc
from sklearn.metrics import accuracy_score, f1_score


########################

def read_images_from_folder(folder_path):
    images = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):  # Check for common image file extensions
            img_path = os.path.join(folder_path, filename)

            with Image.open(img_path) as img:
                img_array = np.array(img, dtype=np.uint8)  # Convert image to NumPy array with float32here, and the code be
                images.append(img_array)  # Add the image array to the list

    images_np = np.stack(images)  # Convert the list of image arrays to a single NumPy array
    return images_np



def evaluate_pipeline(model, X_test_raw, y_test, preprocessing_fn):
    """
    Evaluate a machine learning pipeline.

    Parameters:
    - model: Trained machine learning model.
    - X_test_raw: Raw test data.
    - y_test: True labels for test data.
    - preprocessing_fn: Function to preprocess raw data.

    Returns:
    - metrics: Dictionary containing evaluation metrics.
    """
    # Lists to store memory and CPU usage data
    mem_usage = []
    cpu_usage = []

    # Event to stop monitoring
    stop_monitoring = threading.Event()

    # Function to monitor resources
    def monitor():
        process = psutil.Process(os.getpid())
        while not stop_monitoring.is_set():
            mem = process.memory_info().rss / (1024 * 1024)  # Memory in MB
            cpu = process.cpu_percent(interval=None)  # CPU usage percentage
            mem_usage.append(mem)
            cpu_usage.append(cpu)
            time.sleep(0.1)  # Sampling interval

    # Start resource monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    # Start timing
    start_time = time.time()

    # Preprocess the test data
    X_test_processed = preprocessing_fn(X_test_raw)

    # Make predictions
    y_pred = None
    y_pred = model.predict(X_test_processed)

    # End timing
    end_time = time.time()

    # Stop monitoring
    stop_monitoring.set()
    monitor_thread.join()

    # Collect metrics
    metrics = {}
    metrics['evaluation_time'] = end_time - start_time  # In seconds

    # Compute peak memory usage
    metrics['peak_memory_usage'] = max(mem_usage)  # In MB
    # Compute average CPU usage
    metrics['average_cpu_usage'] = np.mean(cpu_usage)  # In percentage

    # Compute accuracy and F1 score
    if len(y_pred.shape) != 1:
        # from keras.utils import to_categorical
        # y_test = to_categorical(y_test, num_classes=5)
        y_pred = (y_pred > 0.5).astype(int)

    print(y_test.shape)
    print(y_pred.shape)
    
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')

    # Calculate pipeline size
    # Serialize model and preprocessing function
    model_size = len(pickle.dumps(model))
    preprocessing_size = len(pickle.dumps(preprocessing_fn))
    metrics['pipeline_size'] = (model_size + preprocessing_size) / (1024 * 1024)  # In MB

    # Clean up to free memory
    del X_test_processed, y_pred
    gc.collect()

    return metrics


def evaluate_pipeline(model, X_test_raw, y_test, preprocessing_fn):
    """
    Evaluate a machine learning pipeline.

    Parameters:
    - model: Trained machine learning model.
    - X_test_raw: Raw test data.
    - y_test: True labels for test data.
    - preprocessing_fn: Function to preprocess raw data.

    Returns:
    - metrics: Dictionary containing evaluation metrics.
    """
    # Lists to store memory and CPU usage data
    mem_usage = []
    cpu_usage = []

    # Event to stop monitoring
    stop_monitoring = threading.Event()

    # Function to monitor resources
    def monitor():
        process = psutil.Process(os.getpid())
        while not stop_monitoring.is_set():
            mem = process.memory_info().rss / (1024 * 1024)  # Memory in MB
            cpu = process.cpu_percent(interval=None)  # CPU usage percentage
            mem_usage.append(mem)
            cpu_usage.append(cpu)
            time.sleep(0.1)  # Sampling interval

    # Start resource monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    # Start timing
    start_time = time.time()

    # Preprocess the test data
    # Function to preprocess images in parallel. This is to speed up the preprocessing process.
        
    X_test_processed = preprocessing_fn(X_test_raw)

    # Make predictions
    y_pred = None
    y_pred = model.predict(X_test_processed)

    # End timing
    end_time = time.time()

    # Stop monitoring
    stop_monitoring.set()
    monitor_thread.join()

    # Collect metrics
    metrics = {}
    metrics['evaluation_time'] = end_time - start_time  # In seconds

    # Compute peak memory usage
    metrics['peak_memory_usage'] = max(mem_usage)  # In MB
    # Compute average CPU usage
    metrics['average_cpu_usage'] = np.mean(cpu_usage)  # In percentage

    # Compute accuracy and F1 score
    if len(y_pred.shape) != 1:
        # from keras.utils import to_categorical
        # y_test = to_categorical(y_test, num_classes=5)
        y_pred = (y_pred > 0.5).astype(int)

    print(y_test.shape)
    print(y_pred.shape)
    
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')

    # Calculate pipeline size
    # Serialize model and preprocessing function
    model_size = len(pickle.dumps(model))
    preprocessing_size = len(pickle.dumps(preprocessing_fn))
    metrics['pipeline_size'] = (model_size + preprocessing_size) / (1024 * 1024)  # In MB

    # Clean up to free memory
    del X_test_processed, y_pred
    gc.collect()

    return metrics

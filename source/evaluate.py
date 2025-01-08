import threading
import psutil
import os
import time
import numpy as np
import gc
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Function to monitor memory and CPU usage
def monitor_resources(mem_usage, cpu_usage, stop_event):
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        mem = process.memory_info().rss / (1024 * 1024)  # Memory in MB
        cpu = process.cpu_percent(interval=None)  # CPU usage percentage
        mem_usage.append(mem)
        cpu_usage.append(cpu)
        time.sleep(0.1)  # Sampling interval

# Function to preprocess test data
def preprocess_data(preprocessing_fn, X_test_raw):
    return preprocessing_fn(X_test_raw)

# Function to make predictions
def make_predictions(model, X_test_processed):
    return model.predict(X_test_processed)

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, class_names):
    """
    Plot a confusion matrix with labels.

    Parameters:
    - cm: Confusion matrix.
    - class_names: List of class names.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


def print_evaluation_results(metrics, class_names):
    """
    Print evaluation metrics and display the confusion matrix.

    Parameters:
    - metrics: Dictionary containing evaluation metrics.
    - class_names: List of class names for the confusion matrix.
    """
    # Print general metrics in a formatted style
    
    print("\n### Evaluation Metrics ###\n")
    print(f"Evaluation Time:       {metrics['evaluation_time']:.2f} seconds (The time it took for the pipeline to preprocess data and make predictions.)")
    print(f"Peak Memory Usage:     {metrics['peak_memory_usage']:.2f} MB (The maximum memory used during evaluation.)")
    print(f"Average CPU Usage:     {metrics['average_cpu_usage']:.2f} % (The % shows how much of one CPU core was used during the evaluation.)")
    print(f"Algorithm code size:         {metrics['algorithm_code_size']:.2f} MB (The size of the trained model and preprocessing function.)")
    print(f"Accuracy:              {metrics['accuracy']:.3f} (The percentage of correctly classified samples.)")
    print(f"F1 Score:              {metrics['f1_score']:.3f} (A balance of precision and recall, useful for imbalanced datasets.)")

    # Plot the confusion matrix
    print("\n### Confusion Matrix ###\n")
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)

    
# Function to compute evaluation metrics
def compute_metrics(y_test, y_pred, class_names):
    metrics = {}
    if len(y_pred.shape) != 1:
        y_pred = (y_pred > 0.5).astype(int)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)

        # print(y_pred)
        # print(y_test)
        

    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm

    return metrics

# Function to calculate algorithm_code_size
def calculate_algorithmCode_size(model, preprocessing_fn):
    model_size = len(pickle.dumps(model))
    preprocessing_size = len(pickle.dumps(preprocessing_fn))
    return (model_size + preprocessing_size) / (1024 * 1024)  # In MB

# Main function to evaluate the pipeline
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
    class_names = ["Blurry", "Corrupt", "Missing_Data", "Noisy", "Priority"]
    
    # Set CPU affinity to core 4 (the fourth core)
    p = psutil.Process(os.getpid())
    p.cpu_affinity([4])

    # Lists to store memory and CPU usage data
    mem_usage = []
    cpu_usage = []

    # Event to stop monitoring
    stop_monitoring = threading.Event()

    # Start resource monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_resources, args=(mem_usage, cpu_usage, stop_monitoring))
    monitor_thread.start()

    # Start timing
    start_time = time.time()

    # Preprocess the test data
    X_test_processed = preprocess_data(preprocessing_fn, X_test_raw)

    # Make predictions
    y_pred = make_predictions(model, X_test_processed)

    # End timing
    end_time = time.time()

    # Stop monitoring
    stop_monitoring.set()
    monitor_thread.join()

    # Collect metrics
    metrics = {}
    metrics['evaluation_time'] = end_time - start_time  # In seconds
    metrics['peak_memory_usage'] = max(mem_usage)  # In MB
    metrics['average_cpu_usage'] = np.mean(cpu_usage)  # In percentage
    metrics.update(compute_metrics(y_test, y_pred, class_names))  # Add accuracy, F1 score, etc.
    metrics['algorithm_code_size'] = calculate_algorithmCode_size(model, preprocessing_fn)  # In MB

    print_evaluation_results(metrics, class_names)
    
    # Clean up to free memory
    del X_test_processed, y_pred
    gc.collect()

    return metrics
"""
Utility functions for visualizing analysis results
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from PIL import Image

def plot_metrics(train_history, val_history, metric_name, output_path):
    """
    Plot training metrics
    
    Parameters:
    -----------
    train_history : list
        Metric history for training set
    val_history : list
        Metric history for validation set
    metric_name : str
        Name of the metric (e.g. 'Loss', 'Accuracy')
    output_path : str
        Output file path
    """
    # Debug output path
    print(f"Saving {metric_name} plot to: {output_path}")
    
    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label='Train')
    plt.plot(val_history, label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title(f"Training and Validation {metric_name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {metric_name} plot: {output_path}")

def plot_classification_distribution(results, output_path, class_names=None):
    """
    Plot classification result distribution
    
    Parameters:
    -----------
    results : list of dict
        Classification results
    output_path : str
        Output file path
    class_names : list or None
        List of class names (default: ["Positive", "Negative", "Uncertain"])
    """
    if class_names is None:
        class_names = ["Positive", "Negative", "Uncertain"]
    
    # Count aggregation
    counts = [
        sum(1 for r in results if r['prediction'] == class_names[0] and r['uncertainty_score'] < 0.25),
        sum(1 for r in results if r['prediction'] == class_names[1] and r['uncertainty_score'] < 0.25),
        sum(1 for r in results if r['uncertainty_score'] >= 0.25)
    ]
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(counts, labels=class_names, autopct='%1.1f%%', startangle=90, 
            colors=['#4285F4', '#EA4335', '#FBBC05'], shadow=True, 
            explode=(0.05, 0.05, 0.1), textprops={'fontsize': 14})
    plt.title("Classification Distribution", fontsize=16)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved classification distribution plot: {output_path}")
    
def plot_uncertainty_analysis(results, output_dir, threshold=0.25):
    """
    Plot uncertainty analysis results
    """
    # Create dataframe (without pandas)
    positive_scores = [r['positive_score'] for r in results]
    negative_scores = [r['negative_score'] for r in results]
    uncertainty_scores = [r['uncertainty_score'] for r in results]
    predictions = [r['prediction'] for r in results]
    high_uncertainty = [1 if s >= threshold else 0 for s in uncertainty_scores]
    
    # Uncertainty distribution
    plt.figure(figsize=(12, 10))
    
    # Histogram of uncertainty scores
    plt.subplot(2, 2, 1)
    sns.histplot(uncertainty_scores, bins=20, kde=True)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.title("Uncertainty Score Distribution")
    plt.xlabel("Uncertainty Score")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Scatter plot of Positive score vs uncertainty score
    plt.subplot(2, 2, 2)
    colors = ['blue' if p == 'Positive' else 'red' for p in predictions]
    plt.scatter(positive_scores, uncertainty_scores, c=colors, alpha=0.7)
    plt.axhline(threshold, color='r', linestyle='--')
    plt.title("Positive Score vs Uncertainty")
    plt.xlabel("Positive Score")
    plt.ylabel("Uncertainty Score")
    
    # Scatter plot of Negative score vs uncertainty score
    plt.subplot(2, 2, 3)
    plt.scatter(negative_scores, uncertainty_scores, c=colors, alpha=0.7)
    plt.axhline(threshold, color='r', linestyle='--')
    plt.title("Negative Score vs Uncertainty")
    plt.xlabel("Negative Score")
    plt.ylabel("Uncertainty Score")
    
    # Scatter plot of Positive score vs Negative score (colored by uncertainty)
    plt.subplot(2, 2, 4)
    uncertainty_colors = ['orange' if h else 'green' for h in high_uncertainty]
    plt.scatter(positive_scores, negative_scores, c=uncertainty_colors, alpha=0.7)
    plt.title("Positive Score vs Negative Score")
    plt.xlabel("Positive Score")
    plt.ylabel("Negative Score")
    
    plt.tight_layout()
    uncertainty_plot_path = os.path.join(output_dir, 'uncertainty_analysis.png')
    plt.savefig(uncertainty_plot_path)
    plt.close()
    print(f"Saved uncertainty analysis plot: {uncertainty_plot_path}")

def visualize_feature_clusters(features, labels, output_path):
    """
    Visualize feature clustering results
    
    Parameters:
    -----------
    features : numpy.ndarray
        Feature data
    labels : numpy.ndarray
        Cluster labels
    output_path : str
        Output file path
    """
    # Dimension reduction with t-SNE
    # Adjust perplexity based on sample size
    n_samples = features.shape[0]
    perplexity = min(n_samples - 1, 30)  # Use smaller perplexity for small datasets
    
    # Use PCA for very small datasets
    if n_samples <= 10:
        print(f"Small sample size detected ({n_samples}). Using PCA instead of t-SNE for dimension reduction.")
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
    else:
        print(f"Using t-SNE with perplexity={perplexity} for {n_samples} samples.")
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    
    # Apply dimension reduction
    features_2d = reducer.fit_transform(features)
    
    # Plot scatter by clusters with different colors
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=f'Cluster {label}', alpha=0.7)
    
    plt.title("Feature Clusters (Dimension Reduction)", fontsize=16)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved feature clusters plot: {output_path}")

def plot_integrated_heatmap(heatmaps, output_dir):
    """
    Integrate and visualize multiple heatmaps
    
    Parameters:
    -----------
    heatmaps : list of numpy.ndarray
        List of heatmaps
    output_dir : str
        Output directory
    """
    if not heatmaps:
        print("No heatmaps to integrate.")
        return
    
    # Create average and maximum heatmaps
    avg_heatmap = np.mean(heatmaps, axis=0)
    max_heatmap = np.max(heatmaps, axis=0)
    
    plt.figure(figsize=(12, 5))
    
    # Average heatmap
    plt.subplot(1, 2, 1)
    plt.imshow(avg_heatmap, cmap='jet')
    plt.title("Average Heatmap")
    plt.colorbar(label='Activation')
    plt.axis('off')
    
    # Maximum heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(max_heatmap, cmap='jet')
    plt.title("Maximum Heatmap")
    plt.colorbar(label='Activation')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    integrated_path = os.path.join(output_dir, 'integrated_heatmaps.png')
    
    # Print debug info
    print(f"Saving integrated heatmap to: {integrated_path}")
    
    # Save the figure
    plt.savefig(integrated_path)
    plt.close()
    print(f"Saved integrated heatmaps: {integrated_path}")
    
def visualize_example_images(image_paths, predictions, uncertainties, output_dir, max_images=10):
    """
    Visualize sample images with prediction results
    
    Parameters:
    -----------
    image_paths : list
        List of image paths
    predictions : list
        List of prediction results ('Positive' or 'Negative')
    uncertainties : list
        List of uncertainty scores
    output_dir : str
        Output directory
    max_images : int
        Maximum number of images to display
    """
    # Sort by high uncertainty
    sorted_indices = sorted(range(len(uncertainties)), key=lambda i: uncertainties[i], reverse=True)
    selected_indices = sorted_indices[:max_images]
    
    num_rows = min(len(selected_indices), 5)
    num_cols = (len(selected_indices) + num_rows - 1) // num_rows
    
    plt.figure(figsize=(12, 2.5 * num_rows))
    plt.subplots_adjust(hspace=0.4)
    
    for i, idx in enumerate(selected_indices):
        img_path = image_paths[idx]
        pred = predictions[idx]
        uncertainty = uncertainties[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img)
        plt.title(f"{os.path.basename(img_path)}\n{pred} (Uncertainty: {uncertainty:.3f})", fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    examples_path = os.path.join(output_dir, 'example_images.png')
    plt.savefig(examples_path)
    plt.close()
    print(f"Saved example images: {examples_path}")

# Add this function to visualization.py
def plot_classification_histogram(results, output_path, metric_name="Absolute Difference"):
    """
    Plot a histogram of absolute differences between classification results

    Parameters:
    -----------
    results : list of dict
        List of classification results
    output_path : str
        Path of the output file
    metric_name : str
        Name of the metric to be drawn
    """
    # Calculate absolute difference
    abs_differences = []
    for r in results:
        if r['prediction'] == 'Positive':
            abs_diff = abs(r['positive_score'] - r['negative_score'])
            abs_differences.append(abs_diff)
    
    if not abs_differences:  
        print("No Positive classifications found for histogram.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(abs_differences, bins=20, alpha=0.7, color='#4285F4')
    plt.title(f"Histogram of {metric_name} for Positive Classifications")
    plt.xlabel(f"{metric_name} in Softmax Output Probabilities")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved classification histogram: {output_path}")
"""
Model analysis and evaluation module
Includes forecasting, evaluation and uncertainty analysis
"""
import os
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from model_utils import load_model, generate_gradcam, extract_features, calculate_weights, apply_weights
from data_utils import prepare_eval_data, overlay_heatmap, ensure_output_dirs
from visualization import (
    plot_classification_distribution, plot_classification_histogram, plot_uncertainty_analysis, 
    visualize_feature_clusters, plot_integrated_heatmap, visualize_example_images
)


class VirusAnalyzer:
    """
    Class for analyzing and evaluating virus classification models
    """
    def __init__(self, model_path, input_dir, output_dir, uncertainty_threshold=0.25, class_names=None):
        """
        Parameters:
        -----------
        model_path : str
            Path to the model
        input_dir : str
            Directory of input images
        output_dir : str
            Output directory
        uncertainty_threshold : float
            Threshold to determine "high" uncertainty
        class_names : list or None
            List of class names (default: ["Positive", "Negative", "Uncertain"])
        """
        self.model_path = model_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.uncertainty_threshold = uncertainty_threshold
        self.class_names = class_names or ["Positive", "Negative", "Uncertain"]
        
        # Prepare output directories
        ensure_output_dirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'uncertainty'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'uncertainty', 'heatmaps'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'uncertainty', 'clusters'), exist_ok=True)
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model, self.target_layer = load_model(model_path, self.device)
        
        # Calculate weights
        self.weights = calculate_weights()
    
    def predict(self, include_uncertainty_analysis=True):
        """
        Predict images and analyze results
        
        Parameters:
        -----------
        include_uncertainty_analysis : bool
            Whether to include uncertainty analysis
            
        Returns:
        --------
        results : list of dict
            Prediction results
        """
        # Prepare data
        eval_dataset, image_paths = prepare_eval_data(self.input_dir)
        dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=False)
        
        # List to store results
        results = []
        
        # Data for uncertainty analysis
        high_uncertainty_features = []
        high_uncertainty_heatmaps = []
        high_uncertainty_paths = []
        
        # Prepare CSV file
        csv_path = os.path.join(os.path.abspath(self.output_dir), 'sheet', 'prediction_results.csv')
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Filename", "Prediction", f"{self.class_names[0]} Score", f"{self.class_names[1]} Score", 
                f"{self.class_names[2]} Score", "High Uncertainty"
            ])
        
        print(f"Analyzing {len(image_paths)} images...")
        
        # Batch processing
        for batch_idx, (inputs, paths) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            
            # Prediction
            with torch.no_grad():
                outputs = self.model(inputs)
                scores = outputs.clone()
                weighted_scores = apply_weights(scores, self.weights)
            
            # Process each image
            for i, path in enumerate(paths):
                # Get scores
                positive_score, negative_score, uncertainty_score = weighted_scores[i].cpu().numpy()
                
                # Determine prediction class
                prediction = self.class_names[0] if positive_score >= negative_score else self.class_names[1]
                
                # Check uncertainty
                is_high_uncertainty = uncertainty_score >= self.uncertainty_threshold
                
                # Save result
                result = {
                    'filename': os.path.basename(path),
                    'path': path,
                    'prediction': prediction,
                    'positive_score': float(positive_score),
                    'negative_score': float(negative_score),
                    'uncertainty_score': float(uncertainty_score),
                    'high_uncertainty': bool(is_high_uncertainty)
                }
                results.append(result)
                
                # Save to CSV
                with open(csv_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        os.path.basename(path), prediction, 
                        positive_score, negative_score, uncertainty_score,
                        "Yes" if is_high_uncertainty else "No"
                    ])
                
                # Process high uncertainty images
                if include_uncertainty_analysis and is_high_uncertainty:
                    # Generate Grad-CAM (for uncertainty class)
                    gradcam = generate_gradcam(
                        self.model, self.target_layer, inputs[i:i+1], class_idx=2
                    )
                    
                    # Extract features
                    features = extract_features(self.model, inputs[i:i+1])
                    
                    # Overlay heatmap
                    output_path = os.path.join(
                        self.output_dir, 'uncertainty', 'heatmaps', 
                        f"uncertainty_{os.path.basename(path)}"
                    )
                    overlay_heatmap(path, gradcam, output_path)
                    
                    # Save data for uncertainty analysis
                    high_uncertainty_features.append(features)
                    high_uncertainty_heatmaps.append(gradcam)
                    high_uncertainty_paths.append(path)
            
            # Show progress
            print(f"Processed batch {batch_idx+1}/{len(dataloader)}")
        
        print(f"Analysis complete. Results saved to {csv_path}")
        
        # Plot classification distribution
        plot_classification_distribution(
            results, os.path.join(self.output_dir, 'graph', 'classification_distribution.png'),
            class_names=self.class_names
        )
        
        plot_classification_histogram(
            results, os.path.join(self.output_dir, 'graph', 'classification_histogram.png'),
            metric_name="Absolute Differences"
        )
        
        # Uncertainty analysis
        if include_uncertainty_analysis:
            self._analyze_uncertainty(
                results, high_uncertainty_features, high_uncertainty_paths, high_uncertainty_heatmaps
            )
        
        return results
    
    def _analyze_uncertainty(self, results, features, paths, heatmaps):
        """
        Perform detailed uncertainty analysis
        
        Parameters:
        -----------
        results : list of dict
            Prediction results
        features : list of numpy.ndarray
            Features of high uncertainty images
        paths : list of str
            Paths of high uncertainty images
        heatmaps : list of numpy.ndarray
            Heatmaps of high uncertainty images
        """
        if not features:
            print("No high uncertainty images found.")
            return
        
        print(f"Performing detailed uncertainty analysis for {len(features)} images...")
        
        # Plot uncertainty distribution
        plot_uncertainty_analysis(
            results, os.path.join(self.output_dir, 'uncertainty'),
            threshold=self.uncertainty_threshold
        )
        
        # Convert features to NumPy array
        features_array = np.array(features)
        
        # Visualize sample images
        predictions = [r['prediction'] for r in results if r['high_uncertainty']]
        uncertainties = [r['uncertainty_score'] for r in results if r['high_uncertainty']]
        visualize_example_images(
            paths, predictions, uncertainties,
            os.path.join(self.output_dir, 'uncertainty')
        )
        
        # K-means clustering
        # Need at least 3 samples for clustering, and should have more samples than clusters
        if len(features) >= 3:  
            # Adjust n_clusters based on sample size
            n_samples = len(features)
            n_clusters = min(n_samples - 1, 5)  # Ensure n_clusters < n_samples
            
            # Skip clustering if n_clusters is too small
            if n_clusters < 2:
                print(f"Too few samples ({n_samples}) for meaningful clustering. Skipping clustering analysis.")
            else:
                print(f"Performing K-means clustering with {n_clusters} clusters for {n_samples} samples.")
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(features_array)
                
                # Visualize clustering results
                visualize_feature_clusters(
                    features_array, clusters,
                    os.path.join(self.output_dir, 'uncertainty', 'clusters', 'feature_clusters.png')
                )
                
                # Analyze heatmaps by cluster
                cluster_heatmaps = {i: [] for i in range(n_clusters)}
                for i, cluster_id in enumerate(clusters):
                    cluster_heatmaps[cluster_id].append(heatmaps[i])
                
                # Integrated heatmap for each cluster
                for cluster_id, cluster_heatmaps_list in cluster_heatmaps.items():
                    if cluster_heatmaps_list:
                        plot_integrated_heatmap(
                            cluster_heatmaps_list,
                            os.path.join(self.output_dir, 'uncertainty', 'clusters', f'cluster_{cluster_id}')
                        )
        else:
            print(f"Too few high uncertainty samples ({len(features)}) for clustering analysis. Skipping clustering.")
        
        # Integrate all uncertainty heatmaps
        plot_integrated_heatmap(
            heatmaps, os.path.join(self.output_dir, 'uncertainty')
        )
        
        print("Uncertainty analysis complete.")
    
    def evaluate(self, ground_truth_file=None):
        """
        Evaluate the model (if labeled data is available)
        
        Parameters:
        -----------
        ground_truth_file : str or None
            Path to CSV file with ground truth labels
            
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        # If no ground truth, run prediction only
        if ground_truth_file is None:
            print("No ground truth file provided. Running prediction only.")
            return self.predict()
        
        # TODO: Implement evaluation metrics using ground truth
        # This part needs to be extended based on actual data format
        raise NotImplementedError("Evaluation with ground truth is not implemented yet.")
    
def run_analysis(config):
    """
    Run analysis based on configuration
    
    Parameters:
    -----------
    config : dict
        Analysis configuration
    """
    # Extract configuration
    model_path = config.get('model_path', 'model/virus_model.pth')
    input_dir = config.get('input_folder', 'input')
    output_dir = config.get('output_folder', 'output')
    uncertainty_threshold = config.get('uncertainty_threshold', 0.25)
    include_uncertainty = config.get('include_uncertainty_analysis', True)
    class_names = config.get('classes', ["Positive", "Negative", "Uncertain"])
    
    # Check if absolute paths are already set (set in main.py)
    # If not, convert to absolute paths
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    if not os.path.isabs(input_dir):
        input_dir = os.path.abspath(input_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
    
    print("Analysis paths:")
    print(f"  Model path: {model_path}")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create analyzer
    analyzer = VirusAnalyzer(
        model_path=model_path,
        input_dir=input_dir,
        output_dir=output_dir,
        uncertainty_threshold=uncertainty_threshold,
        class_names=class_names
    )
    
    # Run analysis
    results = analyzer.predict(include_uncertainty_analysis=include_uncertainty)
    
    return results
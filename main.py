"""
Main module of the virus image classification system.
"""
import argparse
import yaml
import os
from trainer import train_model
from analyzer import run_analysis

def load_config(config_path="config.yaml"):
    """Read the configuration file"""
    with open(config_path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    """main executable function"""
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description="Virus Image Classification System")
    parser.add_argument('--config', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], help='実行モード')
    args = parser.parse_args()
    
    # Loading the configuration file
    config = load_config(args.config)
    
    # Set mode (preferred if command line arguments are specified).
    mode = args.mode if args.mode else config['mode']
    print(f"Running in {mode} mode.")
    
    # Resolve relative paths with respect to the script's directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.join(script_dir, 'output')
    os.makedirs(output_root, exist_ok=True)
    
    # Path resolution.
    if mode == 'train':
        # Training mode path resolution.
        if 'data_dir' in config['training'] and config['training']['data_dir'].startswith('./'):
            config['training']['data_dir'] = os.path.join(script_dir, config['training']['data_dir'][2:])
        
        # モデルの保存先を設定
        model_dir = os.path.join(script_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        config['training']['model_save_path'] = os.path.join(model_dir, 'virus_model.pth')
        config['training']['output_dir'] = output_root
        
    else:  # predict mode
        # Predictive mode path resolution.
        if 'input_folder' in config['prediction'] and config['prediction']['input_folder'].startswith('./'):
            config['prediction']['input_folder'] = os.path.join(script_dir, config['prediction']['input_folder'][2:])
        
        config['prediction']['model_path'] = os.path.join(script_dir, 'model', 'virus_model.pth')
        config['prediction']['output_folder'] = output_root
    
    # Execution of the process according to the mode.
    if mode == 'train':
        # training mode
        model = train_model(config['training'])
        print("Training completed successfully.")
        
        # Check whether the model is also evaluated after the study
        if config.get('options', {}).get('evaluate_after_training', False):
            print("Evaluating the trained model...")
            prediction_config = config['prediction']
            prediction_config['model_path'] = config['training']['model_save_path']
            prediction_config['output_folder'] = config['training']['output_dir']
            run_analysis(prediction_config)
            
    elif mode == 'predict':
        # Prediction and evaluation mode
        results = run_analysis(config['prediction'])
        print("Prediction and analysis completed successfully.")
        
        # Show summary of results
        if results:
            high_uncertainty_count = sum(1 for r in results if r['high_uncertainty'])
            total_count = len(results)
            class_names = config['prediction'].get('classes', ["Positive", "Negative", "Uncertain"])
            
            print("\nAnalysis Summary:")
            print(f"Total images analyzed: {total_count}")
            print(f"Images with high uncertainty: {high_uncertainty_count} ({high_uncertainty_count/total_count*100:.2f}%)")
            positive_count = sum(1 for r in results if r['prediction'] == class_names[0] and not r['high_uncertainty'])
            negative_count = sum(1 for r in results if r['prediction'] == class_names[1] and not r['high_uncertainty'])
            print(f"{class_names[0]} classifications: {positive_count} ({positive_count/total_count*100:.2f}%)")
            print(f"{class_names[1]} classifications: {negative_count} ({negative_count/total_count*100:.2f}%)")
            print("\nDetailed results are available in the output directory.")
    
    else:
        print(f"Error: Unknown mode '{mode}'. Use 'train' or 'predict'.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze Ray Tune results from hyperparameter optimization.

This script helps parse, analyze, and visualize the results of
Ray Tune experiments for the RecurrentPPO trading agent.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any, Tuple
import warnings

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Plotting libraries not installed. Install with 'pip install matplotlib pandas numpy'")

# Try to import Ray
try:
    import ray
    from ray import tune
    from ray.tune import ExperimentAnalysis
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    warnings.warn("Ray Tune not available. Install with 'pip install ray[tune]'")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Ray Tune results from hyperparameter optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--exp_dir", type=str, required=True,
        help="Directory containing the Ray Tune experiment results"
    )
    parser.add_argument(
        "--metric", type=str, default="eval/mean_reward",
        help="Primary metric to analyze"
    )
    parser.add_argument(
        "--mode", type=str, default="max", choices=["max", "min"],
        help="Whether to maximize or minimize the metric"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save plots and analysis (default: exp_dir/analysis)"
    )
    parser.add_argument(
        "--show_plots", action="store_true",
        help="Show plots interactively (in addition to saving)"
    )
    
    return parser.parse_args()

def load_experiment_data(exp_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load experiment data from directory.
    
    Args:
        exp_dir: Path to the experiment directory
        
    Returns:
        Tuple containing (best_config, all_trial_data)
    """
    if not RAY_AVAILABLE:
        # Fallback to manual loading if Ray is not available
        print("Ray Tune not available, attempting manual loading...")
        best_config_path = os.path.join(exp_dir, "best_config.json")
        
        if os.path.exists(best_config_path):
            with open(best_config_path, 'r') as f:
                best_config = json.load(f)
            print(f"Loaded best config from {best_config_path}")
        else:
            best_config = {}
            print(f"No best_config.json found in {exp_dir}")
        
        # Try to find trial directories
        all_trial_data = []
        for root, dirs, files in os.walk(exp_dir):
            if "result.json" in files:
                try:
                    with open(os.path.join(root, "result.json"), 'r') as f:
                        trial_data = json.load(f)
                    
                    # Try to get config
                    config_path = os.path.join(root, "params.json")
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        trial_data["config"] = config
                    
                    all_trial_data.append(trial_data)
                except Exception as e:
                    print(f"Error loading trial data from {root}: {e}")
        
        return best_config, all_trial_data
    
    # Use Ray Tune's ExperimentAnalysis if available
    try:
        analysis = ExperimentAnalysis(exp_dir)
        print(f"Loaded Ray Tune experiment from {exp_dir}")
        
        # Get best config
        best_config = analysis.get_best_config()
        
        # Get all trial data
        all_trial_data = []
        for trial in analysis.trials:
            data = {
                "trial_id": trial.trial_id,
                "config": trial.config,
                "metrics": trial.last_result
            }
            all_trial_data.append(data)
        
        return best_config, all_trial_data
    except Exception as e:
        print(f"Error loading Ray Tune experiment: {e}")
        return {}, []

def analyze_parameter_importance(
    trial_data: List[Dict[str, Any]], 
    metric: str, 
    mode: str
) -> Dict[str, float]:
    """
    Analyze the importance of different parameters.
    
    This is a very simple analysis that just calculates correlation
    between parameters and the target metric.
    
    Args:
        trial_data: List of trial data dictionaries
        metric: Metric to analyze
        mode: Whether to maximize or minimize the metric
        
    Returns:
        Dictionary mapping parameter names to importance scores
    """
    if not HAS_PLOTTING or len(trial_data) < 2:
        return {}
    
    # Extract data for analysis
    data = []
    for trial in trial_data:
        if isinstance(trial.get("metrics"), dict) and metric in trial.get("metrics", {}):
            row = {}
            # Get parameter values
            config = trial.get("config", {})
            for param, value in config.items():
                if isinstance(value, (int, float)):
                    row[param] = value
            
            # Get metric value
            row[metric] = trial["metrics"][metric]
            data.append(row)
    
    if not data:
        return {}
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate correlations
    correlations = {}
    target = df[metric]
    
    for column in df.columns:
        if column != metric and df[column].dtype in [np.float64, np.int64]:
            # Calculate correlation
            correlation = df[column].corr(target)
            # Adjust sign based on mode
            if mode == "min":
                correlation = -correlation
            correlations[column] = abs(correlation)  # Use absolute correlation for importance
    
    return correlations

def plot_parameter_effects(
    trial_data: List[Dict[str, Any]],
    metric: str,
    mode: str,
    output_dir: str,
    show_plots: bool
) -> None:
    """
    Plot the effects of different parameters on the metric.
    
    Args:
        trial_data: List of trial data dictionaries
        metric: Metric to analyze
        mode: Whether to maximize or minimize the metric
        output_dir: Directory to save plots
        show_plots: Whether to show plots interactively
    """
    if not HAS_PLOTTING or len(trial_data) < 3:
        return
    
    # Convert trial data to DataFrame
    rows = []
    for trial in trial_data:
        if isinstance(trial.get("metrics"), dict) and metric in trial.get("metrics", {}):
            row = {}
            # Get parameter values
            config = trial.get("config", {})
            for param, value in config.items():
                # Only include numeric parameters
                if isinstance(value, (int, float)):
                    row[param] = value
            
            # Get metric value
            row[metric] = trial["metrics"][metric]
            
            # Get additional metrics if available
            if "explained_variance" in trial["metrics"]:
                row["explained_variance"] = trial["metrics"]["explained_variance"]
            
            rows.append(row)
    
    if not rows:
        print("No valid data found for plotting")
        return
    
    df = pd.DataFrame(rows)
    
    # Only keep numeric columns with multiple values
    param_columns = [col for col in df.columns if col != metric and 
                     col != "explained_variance" and 
                     df[col].dtype in [np.float64, np.int64] and 
                     df[col].nunique() > 1]
    
    if not param_columns:
        print("No suitable parameters found for plotting")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot individual parameter effects
    for param in param_columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[param], df[metric], alpha=0.7)
        
        # Add trend line if we have enough points
        if len(df) >= 5:
            try:
                z = np.polyfit(df[param], df[metric], 1)
                p = np.poly1d(z)
                plt.plot(df[param], p(df[param]), "r--", alpha=0.8)
            except np.linalg.LinAlgError:
                pass  # Skip trend line if fitting fails
        
        plt.title(f"Effect of {param} on {metric}")
        plt.xlabel(param)
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"param_effect_{param}.png"), dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # Plot explained variance vs mean reward if available
    if "explained_variance" in df.columns:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            df["explained_variance"], 
            df[metric], 
            c=df[param_columns[0]] if param_columns else None,
            alpha=0.7, 
            cmap="viridis"
        )
        
        plt.title(f"Explained Variance vs {metric}")
        plt.xlabel("Explained Variance")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        
        if param_columns:
            plt.colorbar(scatter, label=param_columns[0])
        
        # Add quadrant lines at 0 for explained variance
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "explained_variance_vs_reward.png"), dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # Create a correlation heatmap
    if len(param_columns) > 1:
        plt.figure(figsize=(12, 10))
        corr = df[param_columns + [metric]].corr()
        
        # Use seaborn if available for better heatmaps
        try:
            import seaborn as sns
            sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        except ImportError:
            plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar()
            
            # Add text annotations
            for i in range(len(corr)):
                for j in range(len(corr)):
                    plt.text(j, i, f"{corr.iloc[i, j]:.2f}", 
                             ha="center", va="center", 
                             color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
        
        plt.title("Parameter Correlations")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()

def plot_training_progress(
    trial_data: List[Dict[str, Any]],
    metric: str,
    output_dir: str,
    show_plots: bool
) -> None:
    """
    Plot training progress over time for different trials.
    
    Args:
        trial_data: List of trial data dictionaries
        metric: Metric to analyze
        output_dir: Directory to save plots
        show_plots: Whether to show plots interactively
    """
    if not HAS_PLOTTING or len(trial_data) < 1:
        return
    
    # Assuming trial data contains training curves
    # This requires Ray Tune experiment data in a specific format
    # If not available, this function will not do anything
    
    # Check if we have the right format of data
    has_progress_data = False
    for trial in trial_data:
        if isinstance(trial.get("metrics"), dict) and "training_iteration" in trial["metrics"]:
            has_progress_data = True
            break
    
    if not has_progress_data:
        print("No training progress data found")
        return
    
    # Find the best performing trials
    best_trials = sorted(
        trial_data,
        key=lambda t: t.get("metrics", {}).get(metric, -float('inf')),
        reverse=True
    )[:5]  # Top 5 trials
    
    if not best_trials:
        return
    
    plt.figure(figsize=(12, 6))
    
    for i, trial in enumerate(best_trials):
        # Extract key parameters for legend
        config = trial.get("config", {})
        label_parts = []
        for param in ["learning_rate", "gamma", "n_steps"]:
            if param in config:
                if param == "learning_rate":
                    label_parts.append(f"lr={config[param]:.6f}")
                elif param == "gamma":
                    label_parts.append(f"γ={config[param]:.3f}")
                else:
                    label_parts.append(f"{param}={config[param]}")
        
        label = f"Trial {i+1}: " + ", ".join(label_parts)
        
        # Plot training curve if available
        if "training_iteration" in trial["metrics"] and metric in trial["metrics"]:
            plt.plot([trial["metrics"]["training_iteration"]], 
                     [trial["metrics"][metric]], 
                     'o-', label=label)
    
    plt.title(f"Training Progress: {metric}")
    plt.xlabel("Training Iteration")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"), dpi=300)
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def generate_summary_report(
    best_config: Dict[str, Any],
    trial_data: List[Dict[str, Any]],
    metric: str,
    mode: str,
    output_dir: str
) -> None:
    """
    Generate a summary report of the hyperparameter optimization.
    
    Args:
        best_config: Best found configuration
        trial_data: List of trial data dictionaries
        metric: Metric that was optimized
        mode: Whether metric was maximized or minimized
        output_dir: Directory to save report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort trials by performance
    sorted_trials = sorted(
        trial_data,
        key=lambda t: t.get("metrics", {}).get(metric, -float('inf') if mode == "max" else float('inf')),
        reverse=(mode == "max")
    )
    
    # Write report
    report_path = os.path.join(output_dir, "optimization_report.md")
    with open(report_path, 'w') as f:
        f.write("# Hyperparameter Optimization Report\n\n")
        
        # Best configuration
        f.write("## Best Configuration\n\n")
        if best_config:
            f.write("```json\n")
            f.write(json.dumps(best_config, indent=2))
            f.write("\n```\n\n")
        else:
            f.write("No best configuration found.\n\n")
        
        # Top performing trials
        f.write("## Top Performing Trials\n\n")
        if sorted_trials:
            f.write("| Rank | Trial ID | Performance | Learning Rate | Gamma | n_steps | Explained Variance |\n")
            f.write("|------|----------|-------------|---------------|-------|---------|-------------------|\n")
            
            for i, trial in enumerate(sorted_trials[:10]):  # Top 10
                metrics = trial.get("metrics", {})
                config = trial.get("config", {})
                
                performance = metrics.get(metric, "N/A")
                if isinstance(performance, float):
                    performance = f"{performance:.4f}"
                
                lr = config.get("learning_rate", "N/A")
                if isinstance(lr, float):
                    lr = f"{lr:.6f}"
                
                gamma = config.get("gamma", "N/A")
                if isinstance(gamma, float):
                    gamma = f"{gamma:.4f}"
                
                n_steps = config.get("n_steps", "N/A")
                
                explained_variance = metrics.get("explained_variance", "N/A")
                if isinstance(explained_variance, float):
                    explained_variance = f"{explained_variance:.4f}"
                
                trial_id = trial.get("trial_id", f"Trial_{i}")
                
                f.write(f"| {i+1} | {trial_id} | {performance} | {lr} | {gamma} | {n_steps} | {explained_variance} |\n")
            
            f.write("\n")
        else:
            f.write("No trial data available.\n\n")
        
        # Parameter importance
        if HAS_PLOTTING and len(trial_data) >= 3:
            importances = analyze_parameter_importance(trial_data, metric, mode)
            if importances:
                f.write("## Parameter Importance\n\n")
                f.write("| Parameter | Importance |\n")
                f.write("|-----------|------------|\n")
                
                for param, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"| {param} | {importance:.4f} |\n")
                
                f.write("\n")
                f.write("*Note: Importance is measured by absolute correlation with the target metric.*\n\n")
        
        # Include links to plots
        f.write("## Analysis Plots\n\n")
        f.write("The following plots visualize the relationship between hyperparameters and performance:\n\n")
        
        plot_files = [f for f in os.listdir(output_dir) if f.endswith(".png")]
        for plot_file in plot_files:
            plot_name = plot_file.replace(".png", "").replace("_", " ").title()
            f.write(f"- [{plot_name}]({plot_file})\n")
        
        f.write("\n")
        
        # Command to use best config
        f.write("## Using the Best Configuration\n\n")
        f.write("To train a model with the best configuration, use the following command:\n\n")
        f.write("```bash\n")
        best_config_path = os.path.join(os.path.dirname(output_dir), "best_config.json")
        f.write(f"python rl_agent/train.py --load_config {best_config_path} --data_path <path> [other options]\n")
        f.write("```\n")
    
    print(f"Summary report generated at {report_path}")

def main():
    """Main function to run the analysis."""
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.exp_dir, "analysis")
    
    # Check if output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load experiment data
    print(f"Loading experiment data from {args.exp_dir}...")
    best_config, trial_data = load_experiment_data(args.exp_dir)
    
    if not trial_data:
        print("No trial data found. Make sure the experiment directory is correct.")
        sys.exit(1)
    
    print(f"Loaded data for {len(trial_data)} trials")
    
    # Generate plots
    if HAS_PLOTTING:
        print("Generating parameter effect plots...")
        plot_parameter_effects(
            trial_data=trial_data,
            metric=args.metric,
            mode=args.mode,
            output_dir=args.output_dir,
            show_plots=args.show_plots
        )
        
        print("Generating training progress plots...")
        plot_training_progress(
            trial_data=trial_data,
            metric=args.metric,
            output_dir=args.output_dir,
            show_plots=args.show_plots
        )
    else:
        print("Plotting libraries not available. Install matplotlib, pandas, and numpy to generate plots.")
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(
        best_config=best_config,
        trial_data=trial_data,
        metric=args.metric,
        mode=args.mode,
        output_dir=args.output_dir
    )
    
    print(f"Analysis complete. Results saved to {args.output_dir}")
    
    # Print the best configuration
    if best_config:
        print("\nBest Configuration:")
        for param, value in best_config.items():
            if param in ["learning_rate", "gamma", "n_steps"]:
                print(f"  {param}: {value}")

if __name__ == "__main__":
    main() 
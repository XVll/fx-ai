# utils/model_analysis.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import json
from PIL import Image
import io
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class ModelAnalyzer:
    """
    Utilities for analyzing model behavior, performance patterns,
    and decision-making processes in trading contexts.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            model_config: Dict[str, Any],
            output_dir: str = "analysis",
            device: Optional[Union[str, torch.device]] = None,
            log_to_wandb: bool = True
    ):
        """
        Initialize the model analyzer.

        Args:
            model: PyTorch model to analyze
            model_config: Model configuration
            output_dir: Directory to save analysis outputs
            device: Device to use (default: infer from model)
            log_to_wandb: Whether to log visualizations to W&B
        """
        self.model = model
        self.model_config = model_config
        self.output_dir = output_dir
        self.log_to_wandb = log_to_wandb

        # Set device
        if device is not None:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        else:
            # Infer from model
            self.device = next(model.parameters()).device

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Track visualizations
        self.visualizations = []

    def analyze_attention_patterns(
            self,
            sample_inputs: Dict[str, torch.Tensor],
            layer_indices: Optional[List[int]] = None,
            output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze attention patterns in transformer layers.

        Args:
            sample_inputs: Sample inputs for the model
            layer_indices: Indices of layers to analyze (default: all layers)
            output_path: Path to save visualization (default: auto-generate)

        Returns:
            Dictionary with analysis results
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in sample_inputs.items()}

        # Get attention matrices (this assumes the model has been modified to return attention)
        # For the actual implementation, you'll need to modify the model to return attention matrices
        with torch.no_grad():
            # Forward pass with attention output
            outputs = self.model(inputs)

            # This is a placeholder for actual attention extraction
            # In practice, you'd modify the model's forward method to return attention matrices
            attentions = self._extract_attention_matrices(outputs)

        # Visualize attention matrices
        fig, axes = plt.subplots(
            len(attentions), 1,
            figsize=(12, 6 * len(attentions)),
            squeeze=False
        )

        for i, (layer_name, attn_matrix) in enumerate(attentions.items()):
            ax = axes[i, 0]

            # Plot attention heatmap
            im = ax.imshow(attn_matrix.cpu().numpy(), cmap="viridis")
            ax.set_title(f"{layer_name} Attention")
            fig.colorbar(im, ax=ax)

            # Add labels
            ax.set_xlabel("Key tokens")
            ax.set_ylabel("Query tokens")

        plt.tight_layout()

        # Save visualization
        if output_path is None:
            os.makedirs(os.path.join(self.output_dir, "attention"), exist_ok=True)
            output_path = os.path.join(self.output_dir, "attention", "attention_patterns.png")

        plt.savefig(output_path)

        # Log to W&B if enabled
        if self.log_to_wandb:
            self._log_visualization_to_wandb("attention_patterns", fig, "attention patterns")

        self.visualizations.append({
            "type": "attention",
            "path": output_path,
            "description": "Attention patterns in transformer layers"
        })

        plt.close(fig)

        return {
            "attention_matrices": {k: v.cpu().numpy() for k, v in attentions.items()},
            "visualization_path": output_path
        }

    def analyze_feature_importance(
            self,
            sample_inputs: Dict[str, torch.Tensor],
            feature_names: Optional[Dict[str, List[str]]] = None,
            output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze feature importance using gradient attribution.

        Args:
            sample_inputs: Sample inputs for the model
            feature_names: Names of features for each input type (optional)
            output_path: Path to save visualization (default: auto-generate)

        Returns:
            Dictionary with analysis results
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Move inputs to device and require gradients
        inputs = {}
        for k, v in sample_inputs.items():
            inputs[k] = v.clone().detach().to(self.device).requires_grad_(True)

        # Forward pass and compute gradients
        action_params, value = self.model(inputs)

        # For continuous actions
        if isinstance(action_params, tuple) and len(action_params) == 2:
            mean, _ = action_params

            # Compute gradients with respect to mean action
            mean.sum().backward()
        else:
            # For discrete actions
            logits = action_params

            # Compute gradients with respect to logits
            logits.sum().backward()

        # Extract gradients
        gradients = {}
        importances = {}

        for key, tensor in inputs.items():
            if tensor.grad is not None:
                # Compute feature importance as abs of gradient * input
                importance = torch.abs(tensor.grad * tensor)

                # Average over batch dimension
                if len(importance.shape) > 2:
                    importance = importance.mean(dim=0)

                gradients[key] = tensor.grad.detach().cpu().numpy()
                importances[key] = importance.detach().cpu().numpy()

        # Visualize feature importance
        fig, axes = plt.subplots(
            len(importances), 1,
            figsize=(12, 6 * len(importances)),
            squeeze=False
        )

        for i, (key, importance) in enumerate(importances.items()):
            ax = axes[i, 0]

            if len(importance.shape) == 2:
                # For sequence features, create heatmap
                im = ax.imshow(importance, aspect="auto", cmap="viridis")
                ax.set_title(f"{key} Feature Importance")
                fig.colorbar(im, ax=ax)

                # Add feature names if provided
                if feature_names and key in feature_names:
                    ax.set_yticks(np.arange(len(feature_names[key])))
                    ax.set_yticklabels(feature_names[key])

                ax.set_xlabel("Sequence Position")
                ax.set_ylabel("Feature")
            else:
                # For static features, create bar chart
                feature_indices = np.arange(len(importance))
                ax.bar(feature_indices, importance)
                ax.set_title(f"{key} Feature Importance")

                # Add feature names if provided
                if feature_names and key in feature_names:
                    ax.set_xticks(feature_indices)
                    ax.set_xticklabels(feature_names[key], rotation=45, ha="right")

                ax.set_xlabel("Feature")
                ax.set_ylabel("Importance")

        plt.tight_layout()

        # Save visualization
        if output_path is None:
            os.makedirs(os.path.join(self.output_dir, "feature_importance"), exist_ok=True)
            output_path = os.path.join(self.output_dir, "feature_importance", "feature_importance.png")

        plt.savefig(output_path)

        # Log to W&B if enabled
        if self.log_to_wandb:
            self._log_visualization_to_wandb("feature_importance", fig, "feature importance")

        self.visualizations.append({
            "type": "feature_importance",
            "path": output_path,
            "description": "Feature importance analysis using gradient attribution"
        })

        plt.close(fig)

        return {
            "importances": importances,
            "gradients": gradients,
            "visualization_path": output_path
        }

    def analyze_latent_space(
            self,
            states: List[Dict[str, torch.Tensor]],
            actions: List[torch.Tensor],
            rewards: List[float],
            method: str = "tsne",
            output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze the model's latent space using dimension reduction techniques.

        Args:
            states: List of state dictionaries
            actions: List of action tensors
            rewards: List of reward values
            method: Dimension reduction method ('tsne' or 'pca')
            output_path: Path to save visualization (default: auto-generate)

        Returns:
            Dictionary with analysis results
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Extract latent representations
        latent_vectors = []

        with torch.no_grad():
            for state in states:
                # Move state to device
                state_device = {k: v.to(self.device) for k, v in state.items()}

                # Forward pass to get latent representation
                # This assumes model has been modified to expose the fused representation
                latent = self._extract_latent_representation(state_device)
                latent_vectors.append(latent.cpu().numpy())

        latent_array = np.vstack(latent_vectors)

        # Apply dimension reduction
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
            reduced_vectors = reducer.fit_transform(latent_array)
            title = "t-SNE Visualization of Latent Space"
        else:  # pca
            reducer = PCA(n_components=2, random_state=42)
            reduced_vectors = reducer.fit_transform(latent_array)
            title = "PCA Visualization of Latent Space"

        # Visualize reduced space
        fig, ax = plt.subplots(figsize=(12, 10))

        # Color points by reward
        reward_array = np.array(rewards)
        scatter = ax.scatter(
            reduced_vectors[:, 0],
            reduced_vectors[:, 1],
            c=reward_array,
            cmap="viridis",
            alpha=0.7,
            s=50
        )
        plt.colorbar(scatter, label="Reward")

        # Add labels
        ax.set_title(title)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

        plt.tight_layout()

        # Save visualization
        if output_path is None:
            os.makedirs(os.path.join(self.output_dir, "latent_space"), exist_ok=True)
            output_path = os.path.join(self.output_dir, "latent_space", f"latent_space_{method}.png")

        plt.savefig(output_path)

        # Log to W&B if enabled
        if self.log_to_wandb:
            self._log_visualization_to_wandb(f"latent_space_{method}", fig, f"latent space ({method})")

        self.visualizations.append({
            "type": "latent_space",
            "path": output_path,
            "description": f"Latent space visualization using {method}"
        })

        plt.close(fig)

        return {
            "reduced_vectors": reduced_vectors,
            "latent_vectors": latent_array,
            "rewards": reward_array,
            "visualization_path": output_path
        }

    def analyze_action_patterns(
            self,
            states: List[Dict[str, torch.Tensor]],
            market_data: Optional[pd.DataFrame] = None,
            output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze action patterns in relation to market data.

        Args:
            states: List of state dictionaries
            market_data: Market data corresponding to states (optional)
            output_path: Path to save visualization (default: auto-generate)

        Returns:
            Dictionary with analysis results
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Collect model actions
        actions = []
        probabilities = []
        values = []

        with torch.no_grad():
            for state in states:
                # Move state to device
                state_device = {k: v.to(self.device) for k, v in state.items()}

                # Get action and action info from model
                action, action_info = self.model.get_action(state_device)

                # Extract relevant information
                actions.append(action.cpu().numpy())

                if isinstance(action_info, dict):
                    if 'mean' in action_info:
                        # For continuous actions
                        probabilities.append(action_info['mean'].cpu().numpy())
                    elif 'logits' in action_info:
                        # For discrete actions
                        probs = torch.softmax(action_info['logits'], dim=-1)
                        probabilities.append(probs.cpu().numpy())

                    if 'value' in action_info:
                        values.append(action_info['value'].cpu().numpy())

        # Convert to arrays
        action_array = np.vstack(actions) if actions and len(actions[0].shape) > 0 else np.array(actions)
        prob_array = np.vstack(probabilities) if probabilities else None
        value_array = np.vstack(values) if values else None

        # Create visualizations
        fig, axes = plt.subplots(
            3 if market_data is not None else 2, 1,
            figsize=(12, 15 if market_data is not None else 10),
            sharex=True
        )

        # Plot actions
        axes[0].plot(action_array, label="Actions", color="blue")
        axes[0].set_title("Model Actions")
        axes[0].set_ylabel("Action Value")
        axes[0].grid(True)

        # Plot action probabilities or means
        if prob_array is not None:
            if prob_array.shape[1] > 1:
                # For discrete actions
                for i in range(prob_array.shape[1]):
                    axes[1].plot(prob_array[:, i], label=f"Action {i}")
                axes[1].legend()
                axes[1].set_title("Action Probabilities")
            else:
                # For continuous actions (means)
                axes[1].plot(prob_array, label="Action Mean", color="green")
                axes[1].set_title("Action Means")

            axes[1].set_ylabel("Probability/Mean")
            axes[1].grid(True)

        # If market data available, plot price
        if market_data is not None and 'price' in market_data.columns:
            axes[2].plot(market_data['price'], label="Price", color="red")
            axes[2].set_title("Market Price")
            axes[2].set_ylabel("Price")
            axes[2].set_xlabel("Time Step")
            axes[2].grid(True)

        plt.tight_layout()

        # Save visualization
        if output_path is None:
            os.makedirs(os.path.join(self.output_dir, "action_patterns"), exist_ok=True)
            output_path = os.path.join(self.output_dir, "action_patterns", "action_patterns.png")

        plt.savefig(output_path)

        # Log to W&B if enabled
        if self.log_to_wandb:
            self._log_visualization_to_wandb("action_patterns", fig, "action patterns")

        self.visualizations.append({
            "type": "action_patterns",
            "path": output_path,
            "description": "Analysis of model action patterns"
        })

        plt.close(fig)

        return {
            "actions": action_array,
            "probabilities": prob_array,
            "values": value_array,
            "visualization_path": output_path
        }

    def analyze_market_sensitivity(
            self,
            base_state: Dict[str, torch.Tensor],
            market_feature: str = "price",
            feature_range: Tuple[float, float] = (0.5, 1.5),
            num_steps: int = 50,
            branch: str = "hf_features",
            feature_idx: int = 0,
            output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze model sensitivity to changes in market features.

        Args:
            base_state: Base state dictionary
            market_feature: Name of the market feature to vary
            feature_range: Range of feature values as a multiplier of base value
            num_steps: Number of steps in the feature range
            branch: Branch containing the feature ('hf_features', 'mf_features', etc.)
            feature_idx: Index of the feature in the branch
            output_path: Path to save visualization (default: auto-generate)

        Returns:
            Dictionary with analysis results
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Get base state
        base_state = {k: v.clone().to(self.device) for k, v in base_state.items()}

        # Extract base value for the feature
        if branch not in base_state:
            raise ValueError(f"Branch {branch} not found in state")

        # Get feature values in last position (most recent) for the specified feature
        base_value = base_state[branch][0, -1, feature_idx].item()

        # Generate feature values
        feature_values = np.linspace(
            base_value * feature_range[0],
            base_value * feature_range[1],
            num_steps
        )

        # Collect results
        actions = []
        action_means = []
        action_probs = []
        values = []

        with torch.no_grad():
            for feature_value in feature_values:
                # Clone base state
                state = {k: v.clone() for k, v in base_state.items()}

                # Modify feature value
                state[branch][0, -1, feature_idx] = feature_value

                # Get action and action info
                action, action_info = self.model.get_action(state)

                # Extract relevant information
                actions.append(action.cpu().numpy())

                if isinstance(action_info, dict):
                    if 'mean' in action_info:
                        # For continuous actions
                        action_means.append(action_info['mean'].cpu().numpy())
                    elif 'logits' in action_info:
                        # For discrete actions
                        probs = torch.softmax(action_info['logits'], dim=-1)
                        action_probs.append(probs.cpu().numpy())

                    if 'value' in action_info:
                        values.append(action_info['value'].cpu().numpy())

        # Convert to arrays
        action_array = np.vstack(actions) if actions and len(actions[0].shape) > 0 else np.array(actions)
        if action_means:
            mean_array = np.vstack(action_means)
        if action_probs:
            prob_array = np.vstack(action_probs)
        value_array = np.vstack(values) if values else None

        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot actions
        axes[0].plot(feature_values, action_array, marker='o', label="Action")
        axes[0].set_title(f"Model Sensitivity to {market_feature}")
        axes[0].set_ylabel("Action Value")
        axes[0].axvline(x=base_value, color='r', linestyle='--', label="Base Value")
        axes[0].grid(True)
        axes[0].legend()

        # Plot values
        if value_array is not None:
            axes[1].plot(feature_values, value_array, marker='o', label="Value", color="green")
            axes[1].set_title("Value Function Sensitivity")
            axes[1].set_ylabel("Value")
            axes[1].set_xlabel(f"{market_feature} Value")
            axes[1].axvline(x=base_value, color='r', linestyle='--', label="Base Value")
            axes[1].grid(True)
            axes[1].legend()

        plt.tight_layout()

        # Save visualization
        if output_path is None:
            os.makedirs(os.path.join(self.output_dir, "market_sensitivity"), exist_ok=True)
            output_path = os.path.join(
                self.output_dir,
                "market_sensitivity",
                f"sensitivity_{market_feature}.png"
            )

        plt.savefig(output_path)

        # Log to W&B if enabled
        if self.log_to_wandb:
            self._log_visualization_to_wandb(
                f"market_sensitivity_{market_feature}",
                fig,
                f"market sensitivity to {market_feature}"
            )

        self.visualizations.append({
            "type": "market_sensitivity",
            "path": output_path,
            "description": f"Analysis of model sensitivity to {market_feature}"
        })

        plt.close(fig)

        return {
            "feature": market_feature,
            "feature_values": feature_values,
            "actions": action_array,
            "values": value_array,
            "visualization_path": output_path
        }

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive analysis report.

        Args:
            output_path: Path to save the report (default: auto-generate)

        Returns:
            Path to the generated report
        """
        # Create report timestamp
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Auto-generate output path if not provided
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"model_analysis_report_{timestamp}.html")

        # Create HTML report
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .visualization {{
            margin: 30px 0;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .visualization img {{
            max-width: 100%;
            margin: 10px 0;
        }}
        .model-info {{
            font-family: monospace;
            white-space: pre-wrap;
            background-color: #f8f8f8;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <h1>Model Analysis Report</h1>
    <p>
        <strong>Generated:</strong> {timestamp}<br>
        <strong>Model:</strong> {type(self.model).__name__}
    </p>

    <h2>Model Architecture</h2>
    <div class="model-info">
{str(self.model)}
    </div>

    <h2>Model Configuration</h2>
    <div class="model-info">
{json.dumps(self.model_config, indent=2)}
    </div>

    <h2>Visualizations</h2>
"""

        # Add visualizations
        for viz in self.visualizations:
            html += f"""
    <div class="visualization">
        <h3>{viz["type"].replace("_", " ").title()}</h3>
        <p>{viz["description"]}</p>
        <img src="file://{os.path.abspath(viz["path"])}" alt="{viz["type"]}">
    </div>
"""

        # Close HTML
        html += """
</body>
</html>
"""

        # Save report
        with open(output_path, "w") as f:
            f.write(html)

        return output_path

    def _extract_attention_matrices(self, outputs: Any) -> Dict[str, torch.Tensor]:
        """
        Extract attention matrices from model outputs.
        This is a placeholder method - actual implementation would depend on model architecture.
        """
        # This is a placeholder - actual implementation would require model modification
        return {
            "hf_layer0": torch.rand(10, 10),
            "mf_layer0": torch.rand(10, 10),
            "lf_layer0": torch.rand(10, 10)
        }

    def _extract_latent_representation(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract latent representation from model.
        This is a placeholder method - actual implementation would depend on model architecture.
        """
        # In practice, you'd need to modify the model to expose the fused representation
        # Here we just run forward pass and use the first action parameter as a proxy
        action_params, _ = self.model(state)

        if isinstance(action_params, tuple):
            # For continuous actions
            return action_params[0]  # mean
        else:
            # For discrete actions
            return action_params  # logits

    def _log_visualization_to_wandb(self, name: str, fig: plt.Figure, description: str) -> None:
        """Log a visualization to W&B."""
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({name: wandb.Image(fig, caption=description)})
        except ImportError:
            print("Warning: wandb not installed, skipping logging")
        except Exception as e:
            print(f"Warning: Failed to log visualization to W&B: {str(e)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Model analysis tools")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--config-path", type=str, required=True, help="Path to model config")
    parser.add_argument("--data-path", type=str, help="Path to sample data for analysis")
    parser.add_argument("--output-dir", type=str, default="analysis", help="Output directory")
    parser.add_argument("--analysis-type", type=str, choices=[
        "all", "attention", "feature", "latent", "action", "sensitivity"
    ], default="all", help="Type of analysis to perform")
    parser.add_argument("--log-to-wandb", action="store_true", help="Log visualizations to W&B")

    args = parser.parse_args()

    # Load model and config
    model_state = torch.load(args.model_path, map_location="cpu")

    with open(args.config_path, "r") as f:
        config = json.load(f)

    # Create model instance (example - adjust based on your model)
    from models.transformer import MultiBranchTransformer
    model = MultiBranchTransformer(**config)
    model.load_state_dict(model_state["model_state_dict"])

    # Load sample data if provided
    sample_data = None
    if args.data_path and os.path.exists(args.data_path):
        with open(args.data_path, "r") as f:
            sample_data = json.load(f)

    # Create analyzer
    analyzer = ModelAnalyzer(
        model=model,
        model_config=config,
        output_dir=args.output_dir,
        log_to_wandb=args.log_to_wandb
    )

    # Run analyses based on type
    if args.analysis_type in ["all", "feature"] and sample_data:
        analyzer.analyze_feature_importance(sample_data["sample_inputs"])

    if args.analysis_type in ["all", "action"] and sample_data:
        analyzer.analyze_action_patterns(
            sample_data["states"],
            sample_data.get("market_data")
        )

    if args.analysis_type in ["all", "latent"] and sample_data:
        analyzer.analyze_latent_space(
            sample_data["states"],
            sample_data["actions"],
            sample_data["rewards"]
        )

    if args.analysis_type in ["all", "sensitivity"] and sample_data:
        analyzer.analyze_market_sensitivity(sample_data["sample_inputs"])

    # Generate report
    report_path = analyzer.generate_report()
    print(f"Analysis report generated: {report_path}")


if __name__ == "__main__":
    main()
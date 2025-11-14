import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union

from src.config.controller import Controller
from framework.prelude import Settings, Individual


class ControllerVisualizer:
    """Visualizer for Controller neural network parameters."""

    def __init__(self, controller: Controller):
        """
        Initialize the visualizer with a Controller instance.

        Args:
            controller: Controller instance to visualize
        """
        self.controller = controller
        self.layer_params = self._extract_layer_parameters()

    def _extract_layer_parameters(self) -> list[dict[str, np.ndarray]]:
        """
        Extract weight and bias parameters for each layer.

        Returns:
            List of dicts containing 'weight' and 'bias' arrays for each layer
        """
        layers = []
        for i, module in enumerate(self.controller.sequential):
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.detach().cpu().numpy()
                bias = module.bias.detach().cpu().numpy()
                layers.append({
                    'name': f'Layer {i//2 + 1}',  # Skip activation layers in count
                    'weight': weight,
                    'bias': bias,
                    'shape': f'{weight.shape[1]} → {weight.shape[0]}'
                })
        return layers

    def visualize_weights_heatmap(self,
                                   save_path: Optional[Union[str, Path]] = None,
                                   figsize: tuple[int, int] = (16, 10),
                                   cmap: str = 'RdBu_r') -> plt.Figure:
        """
        Create heatmap visualization of all weight matrices.

        Args:
            save_path: Optional path to save the figure
            figsize: Figure size (width, height)
            cmap: Colormap for heatmaps

        Returns:
            matplotlib Figure object
        """
        n_layers = len(self.layer_params)
        fig, axes = plt.subplots(2, n_layers, figsize=figsize)

        if n_layers == 1:
            axes = axes.reshape(2, 1)

        for i, layer_data in enumerate(self.layer_params):
            # Weight heatmap
            weight = layer_data['weight']
            vmax = np.abs(weight).max()

            im1 = axes[0, i].imshow(weight, cmap=cmap, aspect='auto',
                                     vmin=-vmax, vmax=vmax)
            axes[0, i].set_title(f"{layer_data['name']} Weights\n{layer_data['shape']}",
                                 fontsize=10, fontweight='bold')
            axes[0, i].set_xlabel('Input Dim', fontsize=8)
            axes[0, i].set_ylabel('Output Dim', fontsize=8)
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)

            # Bias visualization as horizontal bar
            bias = layer_data['bias'].reshape(1, -1)
            vmax_bias = np.abs(bias).max()

            im2 = axes[1, i].imshow(bias, cmap=cmap, aspect='auto',
                                     vmin=-vmax_bias, vmax=vmax_bias)
            axes[1, i].set_title(f"{layer_data['name']} Biases",
                                 fontsize=10, fontweight='bold')
            axes[1, i].set_xlabel('Output Dim', fontsize=8)
            axes[1, i].set_yticks([])
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)

        plt.suptitle('Controller Parameter Heatmaps', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved heatmap visualization to {save_path}")

        return fig

    def visualize_parameter_distribution(self,
                                         save_path: Optional[Union[str, Path]] = None,
                                         figsize: tuple[int, int] = (16, 8),
                                         bins: int = 50) -> plt.Figure:
        """
        Create distribution plots for all parameters.

        Args:
            save_path: Optional path to save the figure
            figsize: Figure size (width, height)
            bins: Number of bins for histograms

        Returns:
            matplotlib Figure object
        """
        n_layers = len(self.layer_params)
        fig, axes = plt.subplots(2, n_layers, figsize=figsize)

        if n_layers == 1:
            axes = axes.reshape(2, 1)

        all_weights = []
        all_biases = []

        for i, layer_data in enumerate(self.layer_params):
            # Weight distribution
            weight = layer_data['weight'].flatten()
            all_weights.extend(weight)

            axes[0, i].hist(weight, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
            axes[0, i].set_title(f"{layer_data['name']} Weight Distribution",
                                 fontsize=10, fontweight='bold')
            axes[0, i].set_xlabel('Weight Value', fontsize=8)
            axes[0, i].set_ylabel('Frequency', fontsize=8)
            axes[0, i].axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            axes[0, i].grid(True, alpha=0.3)

            # Add statistics
            mean_w = weight.mean()
            std_w = weight.std()
            axes[0, i].text(0.02, 0.98, f'μ={mean_w:.3f}\nσ={std_w:.3f}',
                           transform=axes[0, i].transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                           fontsize=8)

            # Bias distribution
            bias = layer_data['bias'].flatten()
            all_biases.extend(bias)

            axes[1, i].hist(bias, bins=bins, alpha=0.7, color='coral', edgecolor='black')
            axes[1, i].set_title(f"{layer_data['name']} Bias Distribution",
                                 fontsize=10, fontweight='bold')
            axes[1, i].set_xlabel('Bias Value', fontsize=8)
            axes[1, i].set_ylabel('Frequency', fontsize=8)
            axes[1, i].axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            axes[1, i].grid(True, alpha=0.3)

            # Add statistics
            mean_b = bias.mean()
            std_b = bias.std()
            axes[1, i].text(0.02, 0.98, f'μ={mean_b:.3f}\nσ={std_b:.3f}',
                           transform=axes[1, i].transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                           fontsize=8)

        plt.suptitle('Controller Parameter Distributions', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved distribution visualization to {save_path}")

        return fig

    def visualize_parameter_summary(self,
                                    save_path: Optional[Union[str, Path]] = None,
                                    figsize: tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Create a comprehensive summary visualization.

        Args:
            save_path: Optional path to save the figure
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        # 1. Overall parameter distribution
        ax1 = fig.add_subplot(gs[0, :])
        all_params = np.concatenate([
            layer['weight'].flatten() for layer in self.layer_params
        ] + [
            layer['bias'].flatten() for layer in self.layer_params
        ])
        ax1.hist(all_params, bins=100, alpha=0.7, color='purple', edgecolor='black')
        ax1.set_title('Overall Parameter Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Parameter Value', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax1.grid(True, alpha=0.3)

        # Add statistics
        mean_all = all_params.mean()
        std_all = all_params.std()
        min_all = all_params.min()
        max_all = all_params.max()
        ax1.text(0.02, 0.98,
                f'μ={mean_all:.4f}, σ={std_all:.4f}\nmin={min_all:.4f}, max={max_all:.4f}\nTotal params={len(all_params)}',
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                fontsize=9)

        # 2. Layer-wise statistics (weights)
        ax2 = fig.add_subplot(gs[1, 0])
        layer_names = [layer['name'] for layer in self.layer_params]
        weight_means = [layer['weight'].mean() for layer in self.layer_params]
        weight_stds = [layer['weight'].std() for layer in self.layer_params]

        x = np.arange(len(layer_names))
        ax2.bar(x, weight_means, alpha=0.7, color='steelblue', label='Mean')
        ax2.errorbar(x, weight_means, yerr=weight_stds, fmt='none',
                    ecolor='red', capsize=5, label='Std Dev')
        ax2.set_xticks(x)
        ax2.set_xticklabels(layer_names, rotation=0, fontsize=9)
        ax2.set_title('Weight Statistics by Layer', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Value', fontsize=10)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Layer-wise statistics (biases)
        ax3 = fig.add_subplot(gs[1, 1])
        bias_means = [layer['bias'].mean() for layer in self.layer_params]
        bias_stds = [layer['bias'].std() for layer in self.layer_params]

        ax3.bar(x, bias_means, alpha=0.7, color='coral', label='Mean')
        ax3.errorbar(x, bias_means, yerr=bias_stds, fmt='none',
                    ecolor='red', capsize=5, label='Std Dev')
        ax3.set_xticks(x)
        ax3.set_xticklabels(layer_names, rotation=0, fontsize=9)
        ax3.set_title('Bias Statistics by Layer', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Value', fontsize=10)
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Parameter magnitude heatmap
        ax4 = fig.add_subplot(gs[2, 0])
        param_sizes = [layer['weight'].size + layer['bias'].size for layer in self.layer_params]
        weight_norms = [np.linalg.norm(layer['weight']) for layer in self.layer_params]
        bias_norms = [np.linalg.norm(layer['bias']) for layer in self.layer_params]

        x = np.arange(len(layer_names))
        width = 0.35
        ax4.bar(x - width/2, weight_norms, width, label='Weight L2 Norm', alpha=0.8, color='steelblue')
        ax4.bar(x + width/2, bias_norms, width, label='Bias L2 Norm', alpha=0.8, color='coral')
        ax4.set_xticks(x)
        ax4.set_xticklabels(layer_names, rotation=0, fontsize=9)
        ax4.set_title('Parameter L2 Norms by Layer', fontsize=11, fontweight='bold')
        ax4.set_ylabel('L2 Norm', fontsize=10)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Network architecture visualization
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        # Draw network architecture
        layer_sizes = []
        for layer in self.layer_params:
            weight_shape = layer['weight'].shape
            if not layer_sizes:
                layer_sizes.append(weight_shape[1])  # Input size
            layer_sizes.append(weight_shape[0])  # Output size

        # Simple text-based architecture display
        arch_text = "Network Architecture:\n\n"
        arch_text += f"Input: {layer_sizes[0]} features\n"
        for i, layer in enumerate(self.layer_params):
            arch_text += f"  ↓\n"
            arch_text += f"{layer['name']}: {layer['shape']}\n"
            arch_text += f"  Mish Activation\n"
        arch_text += f"  ↓\n"
        arch_text += f"Output: {layer_sizes[-1]} (sigmoid)\n\n"
        arch_text += f"Total Parameters: {sum(param_sizes)}"

        ax5.text(0.1, 0.9, arch_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.suptitle('Controller Parameter Analysis Summary',
                    fontsize=14, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved summary visualization to {save_path}")

        return fig

    def save_all_visualizations(self, output_dir: Union[str, Path], prefix: str = "controller"):
        """
        Generate and save all visualization types.

        Args:
            output_dir: Directory to save visualizations
            prefix: Prefix for output filenames
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Generating visualizations in {output_path}")

        # Generate all visualizations
        self.visualize_weights_heatmap(
            save_path=output_path / f"{prefix}_heatmap.png"
        )
        plt.close()

        self.visualize_parameter_distribution(
            save_path=output_path / f"{prefix}_distribution.png"
        )
        plt.close()

        self.visualize_parameter_summary(
            save_path=output_path / f"{prefix}_summary.png"
        )
        plt.close()

        print(f"All visualizations saved to {output_path}")


def visualize_controller_from_individual(
    individual: Individual,
    settings: Settings,
    output_dir: Union[str, Path],
    prefix: str = "controller"
) -> ControllerVisualizer:
    """
    Create visualizations from an Individual's parameters.

    Args:
        individual: Individual containing controller parameters
        settings: Settings object for controller initialization
        output_dir: Directory to save visualizations
        prefix: Prefix for output filenames

    Returns:
        ControllerVisualizer instance
    """
    # Create controller from individual
    controller = Controller(settings, parameters=individual)

    # Create visualizer
    visualizer = ControllerVisualizer(controller)

    # Generate all visualizations
    visualizer.save_all_visualizations(output_dir, prefix)

    return visualizer


def visualize_controller_from_file(
    checkpoint_path: Union[str, Path],
    settings: Settings,
    output_dir: Union[str, Path],
    prefix: str = "controller"
) -> ControllerVisualizer:
    """
    Create visualizations from a saved checkpoint file.

    Args:
        checkpoint_path: Path to saved individual/controller file
        settings: Settings object for controller initialization
        output_dir: Directory to save visualizations
        prefix: Prefix for output filenames

    Returns:
        ControllerVisualizer instance
    """
    import pickle

    # Load individual from file
    with open(checkpoint_path, 'rb') as f:
        individual = pickle.load(f)

    return visualize_controller_from_individual(individual, settings, output_dir, prefix)
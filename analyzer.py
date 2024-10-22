import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleMechanisticAnalyzer:
    def __init__(self, 
                 model_name: str = "gpt2",  # Changed to a more accessible model
                 output_dir: str = "analysis_results"):
        """
        Initialize the analyzer with a specified model and output directory.
        
        Args:
            model_name: Name of the pretrained model to use
            output_dir: Directory to save analysis results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f'Loading model: {model_name}')
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
            raise
        
        # Improved device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()

    def validate_input(self, text: str, target_word: str) -> bool:
        """Validate input parameters."""
        if not isinstance(text, str) or not isinstance(target_word, str):
            raise ValueError("Both text and target_word must be strings")
        
        if not text.strip() or not target_word.strip():
            raise ValueError("Text and target_word cannot be empty")
            
        # Check if target word is in vocabulary
        target_tokens = self.tokenizer.encode(target_word)
        if len(target_tokens) == 0:
            raise ValueError(f"Target word '{target_word}' not found in vocabulary")
            
        return True

    def analyze_single_fact(self, text: str, target_word: str) -> Optional[Dict]:
        """Analyze how a single fact is processed through the model."""
        try:
            self.validate_input(text, target_word)
            logger.info(f"Analyzing fact: '{text}' with target: '{target_word}'")
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            target_tokens = self.tokenizer.encode(target_word)
            logger.info(f"Target tokens for '{target_word}': {target_tokens}")
            
            if len(target_tokens) > 1:
                logger.warning(f"Target word '{target_word}' tokenizes into multiple tokens")
            target_id = target_tokens[0]
            
            layer_activations = {}
            
            def hook_fn(layer_idx):
                def hook(module, input, output):
                    layer_activations[layer_idx] = output[0].detach().cpu()
                return hook
            
            hooks = []
            for i, layer in enumerate(self.model.transformer.h):  # Updated for GPT-2 architecture
                hook = layer.register_forward_hook(hook_fn(i))
                hooks.append(hook)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            for hook in hooks:
                hook.remove()
            
            if not layer_activations:
                logger.error("No layer activations were captured")
                return None
            
            results = {
                "layer_stats": self._analyze_layer_activations(layer_activations),
                "neuron_activity": self._analyze_neuron_activity(layer_activations),
                "prediction_score": self._get_prediction_score(outputs, target_id),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return None

    def _analyze_layer_activations(self, layer_activations: Dict) -> Dict:
        """Analyze layer activations with improved error handling and statistics."""
        stats = {}
        try:
            all_activations = []
            
            for activations in layer_activations.values():
                flat_activations = activations.reshape(-1).numpy()
                all_activations.append(flat_activations)
            
            all_activations = np.concatenate(all_activations)
            
            # Use robust statistics
            global_median = np.median(all_activations)
            global_iqr = np.percentile(all_activations, 75) - np.percentile(all_activations, 25)
            
            for layer_idx, activations in layer_activations.items():
                act_numpy = activations.numpy()
                flat_act = act_numpy.reshape(-1)
                
                # Normalize using robust statistics
                norm_activations = (flat_act - global_median) / (global_iqr + 1e-6)
                
                stats[layer_idx] = {
                    "mean": float(np.mean(norm_activations)),
                    "median": float(np.median(norm_activations)),
                    "std": float(np.std(norm_activations)),
                    "iqr": float(np.percentile(norm_activations, 75) - np.percentile(norm_activations, 25)),
                    "max": float(np.max(norm_activations)),
                    "min": float(np.min(norm_activations))
                }
                
                logger.info(f"Layer {layer_idx} - median: {stats[layer_idx]['median']:.3f}, "
                          f"IQR: {stats[layer_idx]['iqr']:.3f}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in layer activation analysis: {e}")
            return {}

    def visualize_results(self, results: Dict, save: bool = True) -> None:
        """Visualize analysis results with option to save plots."""
        if not results:
            logger.warning("No results to visualize")
            return

        try:
            fig = plt.figure(figsize=(12, 10))
            gs = plt.GridSpec(2, 1, height_ratios=[1, 1.5], hspace=0.3)
            
            # Plot 1: Layer Statistics
            self._plot_layer_statistics(fig.add_subplot(gs[0]), results)
            
            # Plot 2: Neuron Activity
            self._plot_neuron_activity(fig.add_subplot(gs[1]), results)
            
            if save:
                timestamp = results.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
                filename = os.path.join(self.output_dir, f"analysis_{timestamp}.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Saved visualization to {filename}")
            
            plt.show()
            plt.close()

        except Exception as e:
            logger.error(f"Error in visualization: {e}")

    def _plot_layer_statistics(self, ax, results: Dict) -> None:
        """Helper method for plotting layer statistics."""
        layers = sorted(list(results["layer_stats"].keys()))
        means = [results["layer_stats"][layer]["mean"] for layer in layers]
        stds = [results["layer_stats"][layer]["std"] for layer in layers]
        
        ax.plot(layers, means, 'b-', linewidth=2, label='Mean Activation')
        ax.fill_between(layers, 
                       [m - s for m, s in zip(means, stds)],
                       [m + s for m, s in zip(means, stds)],
                       alpha=0.3, color='blue', label='±1 std dev')
        
        ax.set_title('Layer Activation Statistics', pad=20, size=12, weight='bold')
        ax.set_xlabel('Layer', size=10)
        ax.set_ylabel('Normalized Activation', size=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    def _plot_neuron_activity(self, ax, results: Dict) -> None:
        """Helper method for plotting neuron activity."""
        layers = sorted(list(results["neuron_activity"].keys()))
        neuron_data = np.zeros((len(layers), 10))
        
        for i, layer in enumerate(layers):
            neuron_data[i] = results["neuron_activity"][layer]["top_activations"]
        
        sns.heatmap(neuron_data.T, 
                   ax=ax,
                   cmap='RdBu_r',
                   center=0,
                   vmin=-5,
                   vmax=5,
                   xticklabels=layers,
                   yticklabels=[f'Neuron {n+1}' for n in range(10)],
                   cbar_kws={'label': 'Normalized Activity'})
        
        ax.set_title('Top 10 Neuron Activities per Layer', pad=20, size=12, weight='bold')
        ax.set_xlabel('Layer', size=10)
        ax.set_ylabel('Neurons', size=10)
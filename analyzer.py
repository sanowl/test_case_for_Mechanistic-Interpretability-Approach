# analyzer.py

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class SimpleMechanisticAnalyzer:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B"):
        print('Loading the model...')
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            raise
        
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()

    def analyze_single_fact(self, text: str, target_word: str) -> dict:
        """Analyze how a single fact is processed through the model."""
        print(f"\nAnalyzing fact: '{text}'")
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        target_tokens = self.tokenizer.encode(target_word)
        print(f"Target tokens for '{target_word}': {target_tokens}")
        if len(target_tokens) > 1:
            print(f"Warning: Target word '{target_word}' tokenizes into multiple tokens.")
        target_id = target_tokens[0]
        
        layer_activations = {}
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                layer_activations[layer_idx] = output[0].detach().cpu()
            return hook
        
        hooks = []
        for i, layer in enumerate(self.model.model.layers):
            hook = layer.register_forward_hook(hook_fn(i))
            hooks.append(hook)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        for hook in hooks:
            hook.remove()
        
        if not layer_activations:
            print("No layer activations were captured. Please check the hook registration.")
            return {}
        
        results = {
            "layer_stats": self._analyze_layer_activations(layer_activations),
            "neuron_activity": self._analyze_neuron_activity(layer_activations),
            "prediction_score": self._get_prediction_score(outputs, target_id)
        }
        
        return results

    def _analyze_layer_activations(self, layer_activations: dict) -> dict:
        stats = {}
        all_activations = []
        
        # Collect all activations for better global statistics
        for activations in layer_activations.values():
            # Flatten activations for better statistical analysis
            flat_activations = activations.reshape(-1).numpy()
            all_activations.append(flat_activations)
        
        # Concatenate all activations
        all_activations = np.concatenate(all_activations)
        
        # Calculate robust global statistics
        global_mean = np.median(all_activations)  # More robust than mean
        global_std = np.percentile(all_activations, 75) - np.percentile(all_activations, 25)
        
        # Process each layer
        for layer_idx, activations in layer_activations.items():
            # Convert to numpy and flatten
            act_numpy = activations.numpy()
            flat_act = act_numpy.reshape(-1)
            
            # Calculate layer statistics
            layer_mean = np.mean(flat_act)
            layer_std = np.std(flat_act)

            norm_activations = (flat_act - global_mean) / (global_std + 1e-6)
            
            stats[layer_idx] = {
                "mean": float(np.mean(norm_activations)),
                "std": float(np.std(norm_activations)),
                "raw_mean": float(layer_mean),
                "raw_std": float(layer_std),
                "max": float(np.max(norm_activations)),
                "min": float(np.min(norm_activations))
            }
            
            print(f"Layer {layer_idx} - mean: {stats[layer_idx]['mean']:.3f}, "
                  f"std: {stats[layer_idx]['std']:.3f}")
        
        return stats

    def _analyze_neuron_activity(self, layer_activations: dict) -> dict:
        neuron_activity = {}
        all_neuron_means = []
        
        for activations in layer_activations.values():
        
            neuron_means = activations.mean(dim=[0, 1]).numpy()
            all_neuron_means.extend(neuron_means)
        
        # Calculate robust statistics
        global_median = np.median(all_neuron_means)
        global_iqr = np.percentile(all_neuron_means, 75) - np.percentile(all_neuron_means, 25)
        
        # Second pass: analyze each layer
        for layer_idx, activations in layer_activations.items():
           
            neuron_means = activations.mean(dim=[0, 1]).numpy()
            
            # Normalize using robust statistics
            norm_means = (neuron_means - global_median) / (global_iqr + 1e-6)
            
            # Find most active neurons (both positive and negative activation)
            importance_scores = np.abs(norm_means)
            top_indices = np.argsort(importance_scores)[-10:]
            
            neuron_activity[layer_idx] = {
                "top_neurons": top_indices.tolist(),
                "top_activations": norm_means[top_indices].tolist(),
                "raw_activations": neuron_means[top_indices].tolist()
            }
            
            print(f"Layer {layer_idx} Top Neurons: {top_indices}")
        
        return neuron_activity
    
    def _get_prediction_score(self, outputs, target_id: int) -> float:
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        prediction_confidence = float(probs[0, target_id].cpu().numpy())
        print(f"Prediction confidence for target ID {target_id}: {prediction_confidence:.3f}")
        return prediction_confidence

    def visualize_results(self, results: dict) -> None:
        if not results:
            print("No results to visualize.")
            return

        fig = plt.figure(figsize=(12, 10))
        gs = plt.GridSpec(2, 1, height_ratios=[1, 1.5], hspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        layers = sorted(list(results["layer_stats"].keys()))
        means = [results["layer_stats"][layer]["mean"] for layer in layers]
        stds = [results["layer_stats"][layer]["std"] for layer in layers]
        
        ax1.plot(layers, means, 'b-', linewidth=2, label='Mean Activation')
        ax1.fill_between(layers, 
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.3, color='blue', label='±1 std dev')
        
        ax1.set_title('Layer Activation Statistics', pad=20, size=12, weight='bold')
        ax1.set_xlabel('Layer', size=10)
        ax1.set_ylabel('Normalized Activation', size=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.2)
        
        # Plot 2: Enhanced Neuron Activity Heatmap
        neuron_data = np.zeros((len(layers), 10))
        for i, layer in enumerate(layers):
            neuron_data[i] = results["neuron_activity"][layer]["top_activations"]
        
        # Use a diverging colormap centered at zero
        sns.heatmap(neuron_data.T, 
                   ax=ax2,
                   cmap='RdBu_r',
                   center=0,
                   vmin=-5,  # Limit the scale for better visualization
                   vmax=5,
                   xticklabels=layers,
                   yticklabels=[f'Neuron {n+1}' for n in range(10)],
                   cbar_kws={'label': 'Normalized Activity'})
        
        # Improve second plot appearance
        ax2.set_title('Top 10 Neuron Activities per Layer', pad=20, size=12, weight='bold')
        ax2.set_xlabel('Layer', size=10)
        ax2.set_ylabel('Neurons', size=10)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\nAnalysis Summary:")
        print(f"Prediction confidence: {results['prediction_score']:.3f}")
        print("\nLayer-wise statistics:")
        for layer in layers:
            stats = results["layer_stats"][layer]
            print(f"Layer {layer:2d}: mean = {stats['mean']:6.3f}, "
                  f"std = {stats['std']:6.3f}, "
                  f"max = {stats['max']:6.3f}")

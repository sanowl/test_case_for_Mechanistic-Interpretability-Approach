import torch
import torch.nn as nn
from transformers import  AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as  sns

class SimpleMechanisticAnalyzer:
    def __init__(self, model_name="gpt"):
        print('loading the model')
        self.model= AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("using the device: {self.device}")
        self.model.to(self.device)

    def analyze_single_fact(self, text: str, target_word: str) -> dict:
        """Analyze how a single fact is processed through the model"""
        print(f"\nAnalyzing fact: '{text}'")
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        target_id = self.tokenizer.encode(target_word)[0]
        
        layer_activations = {}
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                layer_activations[layer_idx] = output.detach()
            return hook
        
        hooks = []
        for i in range(len(self.model.transformer.h)):
            hook = self.model.transformer.h[i].register_forward_hook(hook_fn(i))
            hooks.append(hook)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        for hook in hooks:
            hook.remove()
        
        results = {
            "layer_stats": self._analyze_layer_activations(layer_activations),
            "neuron_activity": self._analyze_neuron_activity(layer_activations),
            "prediction_score": self._get_prediction_score(outputs, target_id)
        }
        
        return results
       
    def _analyze_layer_activations(self, layer_activations: dict) -> dict:
        stats = {}
        for layer_idx, activations in layer_activations.items():
            stats[layer_idx] = {
                "mean": float(activations.mean().cpu().numpy()),
                "max": float(activations.max().cpu().numpy()),
                "std": float(activations.std().cpu().numpy())
            }
        return stats
    def _analyze_neuron_activity(self,layer_activations: dict) -> dict:
        neuron_activity = {}
        for layer_idx, activations in layer_activations.items():
            neuron_means = activations.mean(dim=[0,1]).cpu().numpy
            top_neurons = np.argsort(neuron_means)[-5:]
            neuron_activity[layer_idx]  ={
                "top_neurons": top_neurons.tolist(),
                "top_activations": neuron_means[top_neurons].tolist()
            }
        return neuron_activity
    
    def __get_prediction_socre(self,outputs,target_id:int) -> float:
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits,dim=-1)
        return float(probs[0, target_id].cpu().numpy())
    
    def visualize_results(self, results: dict) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        layers = list(results["layer_stats"].keys())
        means = [stats["mean"] for stats in results["layer_stats"].values()]
        stds = [stats["std"] for stats in results["layer_stats"].values()]
        
        ax1.plot(layers, means, 'b-', label='Mean Activation')
        ax1.fill_between(layers, 
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2)
        ax1.set_title('Layer Activation Statistics')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Activation')
        ax1.legend()
        
        neuron_data = np.zeros((len(layers), 5))
        for i, layer in enumerate(layers):
            neuron_data[i] = results["neuron_activity"][layer]["top_activations"]
        
        sns.heatmap(neuron_data.T, ax=ax2, cmap='viridis')
        ax2.set_title('Top 5 Neuron Activities per Layer')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Top Neurons')
        
        plt.tight_layout()
        plt.show()
        
        print("\nAnalysis Summary:")
        print(f"Prediction confidence: {results['prediction_score']:.3f}")
        print("\nLayer-wise activation means:")
        for layer, stats in results["layer_stats"].items():
            print(f"Layer {layer}: {stats['mean']:.3f}")

# Run a demo
if __name__ == "__main__":
    print("Starting mechanistic analysis demo...")
    
    # Initialize analyzer
    analyzer = SimpleMechanisticAnalyzer()
    
    # Test cases
    test_cases = [
        {
            "text": "Michael Jordan plays",
            "target": "basketball"
        },
        {
            "text": "The capital of France is",
            "target": "Paris"
        }
    ]
    
    # Analyze each test case
    for case in test_cases:
        results = analyzer.analyze_single_fact(case["text"], case["target"])
        analyzer.visualize_results(results)
        print("\n" + "="*50 + "\n")




              
  

               
            
            

               
     




    
        
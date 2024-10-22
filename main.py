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

    def analyze_single_fact(self,text:str,target_word:str) -> dict:
       print(f"\nAnalyzing fact: '{text}'")

       inputs =self.tokenizer(text, return_tensors="pt").to(self.device)
       target_id = self.tokenizer.encode(target_word)[0]

       layer_activations = {}

       def hook_fn(layer_idx):
            def hook(module, input, output):
                layer_activations[layer_idx] = output.detach()
            return hook
       
       hooks = []
       for i in range(len(self.model.transformer.h)):
                hook = self.model.transformer.h[i].register_forward_hook(hook_fn(i))
                hooks.append(hooks)

       with torch.no_grad():
            outputs = self.model(**inputs)
            
               
            
            

               
     




    
        
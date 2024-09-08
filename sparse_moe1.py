import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from pathlib import Path
import os
import gc

class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SwiGLU, self).__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
    
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.silu(gate)

class Router(nn.Module):
    def __init__(self, dim_in, num_experts, dropout_rate=0.1):
        super(Router, self).__init__()
        self.num_experts = num_experts
        self.dropout_rate = dropout_rate
        self.gate = nn.Linear(dim_in, num_experts)
    
    def forward(self, x):
        gate_logits = self.gate(x)
        top2_indices, top2_softmax = torch.topk(gate_logits, k=2, dim=-1)
        
        if self.training:
            dropout_mask = torch.bernoulli((1 - self.dropout_rate) * torch.ones_like(top2_softmax))
            top2_softmax = top2_softmax * dropout_mask
            
        return top2_indices, top2_softmax

class Expert(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Expert, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out)
        self.swiglu = SwiGLU(dim_out, dim_out)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.swiglu(x)
        return x

class SparseMoE(nn.Module):
    def __init__(self, expert_models, num_experts=8, num_layers=4, expert_dropout=0.1):
        super(SparseMoE, self).__init__()
        self.expert_models = nn.ModuleList(expert_models)
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.router_layers = nn.ModuleList(
            [Router(dim_in=80, num_experts=num_experts, dropout_rate=expert_dropout) for _ in range(num_layers)]
        )    

    def forward(self, input_features):
        batch_size, seq_length, feature_dim = input_features.size()
        output = input_features

        for layer_idx in range(self.num_layers):
            top2_indices, top2_softmax = self.router_layers[layer_idx](output)
            layer_output = torch.zeros(batch_size, seq_length, feature_dim, device=input_features.device)

            for i in range(2):
                expert_idx = top2_indices[:, :, i]
                weight = top2_softmax[:, :, i].unsqueeze(-1)

                for expert_num in range(self.num_experts):
                    mask = expert_idx == expert_num

                    if mask.any():
                        expert_input = output[mask]
                        expert_output = self.expert_models[expert_num](expert_input)
                        layer_output[mask] += weight[mask] * expert_output
            
            output = layer_output

        return output

def load_model(model_path):
    try:
        model = whisper.load_model(model_path, device="cpu")
        print(f"Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        return None

def process_audio_with_sparse_moe(sparse_moe, audio_path):
    try:
        # Load audio and extract features
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        
        # Extract mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to("cuda")
        
        # Process with SparseMoE
        with torch.no_grad():
            output = sparse_moe(mel.unsqueeze(0))
        
        return output
    except Exception as e:
        print(f"Error during SparseMoE processing: {str(e)}")
        return None

if __name__ == "__main__":
    audio_file_path = "./converted_audio.wav"
    model_dir = "./openai"

    model_paths = [
        "asr_model.pt",
        "bengal_model.pt",
        "tamil_model.pt"
    ]

    models = []
    for model_path in model_paths:
        full_path = os.path.join(model_dir, model_path)
        model = load_model(full_path)
        if model:
            models.append(model)
        gc.collect()
        torch.cuda.empty_cache()

    if not models:
        print("No models were successfully loaded. Exiting.")
        exit(1)

    # Create SparseMoE instance
    sparse_moe = SparseMoE(expert_models=models, num_experts=len(models), num_layers=4).cuda()

    print("Processing with SparseMoE...")
    output = process_audio_with_sparse_moe(sparse_moe, audio_file_path)
    
    if output is not None:
        print("SparseMoE processing completed successfully.")
        print("Output shape:", output.shape)
        # Here you can add further processing or analysis of the output
    else:
        print("Failed to process with SparseMoE.")

    print("Processing complete.")


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperTokenizer
import torchaudio
import os
from pydub import AudioSegment
import tempfile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CustomWhisperFeatureExtractor:
    def __init__(self):
        self.sampling_rate = 16000
        self.n_fft = 400
        self.hop_length = 160
        self.n_mels = 80

    def __call__(self, audio, sampling_rate=16000, return_tensors="pt"):
        if sampling_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
            audio = resampler(audio)

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )(audio)

        if mel_spec.shape[0] > 1:
            mel_spec = mel_spec.mean(dim=0, keepdim=True)

        log_mel_spec = torch.clamp(mel_spec, min=1e-10).log()

        input_features = log_mel_spec.squeeze(0).transpose(0, 1)

        if return_tensors == "pt":
            input_features = input_features.unsqueeze(0)

        return {"input_features": input_features}

def load_whisper_model(model_name):
    try:
        tokenizer = WhisperTokenizer.from_pretrained(model_name)
        model = WhisperModel.from_pretrained(model_name)
        feature_extractor = CustomWhisperFeatureExtractor()
        return model, tokenizer, feature_extractor
    except Exception as e:
        print(f"Error loading Whisper model {model_name}: {str(e)}")
        raise

model_paths = [
    "models/anuragshas_whisper-small-bn",
    "models/kavyamanohar_whisper-small-malayalam",
    "models/Ranjit_odia_whisper_small_v3.0",
]

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
        self.dropout_rate = dropout_rate  # Dropout rate for expert dropout
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

            # Process each expert individually and apply the gating weights
            for i in range(2):
                expert_idx = top2_indices[:, :, i]
                weight = top2_softmax[:, :, i].unsqueeze(-1)

                # Route to experts and combine the results
                for expert_num in range(self.num_experts):
                    mask = expert_idx == expert_num

                    if mask.any():
                        expert_input = output[mask]
                        expert_output = self.expert_models[expert_num](expert_input)
                        layer_output[mask] += weight[mask] * expert_output
            
            output = layer_output

        return output


def convert_m4a_to_wav(m4a_path):
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_wav.name, format="wav")
    return temp_wav.name

def pad_input_features(input_features, target_length=3000):
    current_length = input_features.size(1)
    if current_length < target_length:
        pad_length = target_length - current_length
        input_features = F.pad(input_features, (0, 0, 0, pad_length), "constant", 0)
    elif current_length > target_length:
        input_features = input_features[:, :target_length, :]
    
    return input_features

def load_and_process_audio(file_path, feature_extractor):
    try:
        if file_path.lower().endswith('.m4a'):
            file_path = convert_m4a_to_wav(file_path)
        
        waveform, sample_rate = torchaudio.load(file_path)
        
        input_features = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt").get("input_features")
        
        input_features = pad_input_features(input_features, target_length=3000)
        
        if file_path.endswith('.wav') and file_path != audio_file_path:
            os.remove(file_path)
        
        return input_features.to("cuda")
    except Exception as e:
        print(f"Error processing audio file {file_path}: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        audio_file_path = "/mnt/sdb/data/transcribe/kb_data_clean_m4a/tamil/train/audio/844424933499133-868-m.m4a"
        
        if not os.path.exists(audio_file_path):
            print(f"Error: The file {audio_file_path} does not exist.")
            exit(1)
        
        models_and_tokenizers = [load_whisper_model(path) for path in model_paths]
        models = [model for model, _, _ in models_and_tokenizers]
        tokenizers = [tokenizer for _, tokenizer, _ in models_and_tokenizers]
        feature_extractors = [feature_extractor for _, _, feature_extractor in models_and_tokenizers]
        input_features = load_and_process_audio(audio_file_path, feature_extractors[0])

        print("Input features shape:", input_features.shape)

        sparse_moe = SparseMoE(expert_models=models, num_experts=len(models), num_layers=4).cuda()
        
        with torch.no_grad():
            combined_logits = sparse_moe(input_features)
        
        print("Combined logits shape:", combined_logits.shape)
        
        probs = F.softmax(combined_logits, dim=-1)
        predicted_ids = torch.argmax(probs, dim=-1)
        print("Predicted IDs shape:", predicted_ids.shape)

        transcription = tokenizers[0].decode(predicted_ids[0], skip_special_tokens=True)
        print("Transcription:", transcription)

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        torch.cuda.empty_cache()

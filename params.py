from num2words import num2words
from transformers import WhisperModel, WhisperTokenizer

model_name = "openai/whisper-small"
tokenizer = WhisperTokenizer.from_pretrained(model_name)
model = WhisperModel.from_pretrained(model_name)

state_dict = model.state_dict()

for key in state_dict.keys():
    print(key)

total_params = sum(p.numel() for p in state_dict.values())
print(total_params)
print(f"Total number of parameters: {num2words(total_params)}")

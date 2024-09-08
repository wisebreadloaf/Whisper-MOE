from transformers import WhisperModel, WhisperTokenizer
import os

'''
1. Trained on whole of kathbath dataset - redudant in case of MOE but still a good choice
2. Trained on Odia dataset
3. Trained on hindi dataset
4. Trained on bengali dataset
5. Trained on malayalam dataset
6. Trained on tamil dataset 
7. Trained on Kannada dataset
8. Trained on Telugu dataset 
'''

model_names = ["Rithik101/WhispASR", "Ranjit/odia_whisper_small_v3.0", "vasista22/whisper-hindi-small", "anuragshas/whisper-small-bn", "kavyamanohar/whisper-small-malayalam", "vasista22/whisper-tamil-small", "vasista22/whisper-kannada-small", "steja/whisper-small-telugu-large-data"]
for model_name in model_names:
    custom_directory = f"./models/{model_name.replace('/', '_')}"
    
    if not os.path.exists(custom_directory):
        os.makedirs(custom_directory)
    
    tokenizer = WhisperTokenizer.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name)
    
    tokenizer.save_pretrained(custom_directory)
    model.save_pretrained(custom_directory)

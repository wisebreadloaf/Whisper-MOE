import os
import subprocess
import whisper
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def transcribe_audio(model, audio_path):
    try:
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return None

def translate_to_english(model, audio_path):
    try:
        result = model.transcribe(audio_path, task="translate")
        return result["text"]
    except Exception as e:
        print(f"Error during translation: {str(e)}")
        return None

def load_model(model_path):
    try:
        model = whisper.load_model(model_path, device="cuda")
        print(f"Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        return None

wav_file_path = "./converted_audio.wav"
model_path = "./openai/asr_model.pt"

print("Attempting to load model...")
loaded_model = load_model(model_path)

if loaded_model:
    print("Transcribing audio...")
    transcription = transcribe_audio(loaded_model, wav_file_path)
    
    if transcription:
        print("Transcription:")
        print(transcription)
    else:
        print("Failed to obtain transcription.")
else:
    print("Failed to load the model.")

translation = translate_to_english(loaded_model, wav_file_path)
print("\nTranslation to English:")
print(translation)

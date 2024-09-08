#!/usr/bin/sh
python convert_hf_to_openai.py --checkpoint models/anuragshas_whisper-small-bn/ --whisper_dump_path openai/bengal_model.pt
python convert_hf_to_openai.py --checkpoint models/kavyamanohar_whisper-small-malayalam/ --whisper_dump_path openai/malayalam_model.pt
python convert_hf_to_openai.py --checkpoint models/Ranjit_odia_whisper_small_v3.0/ --whisper_dump_path openai/odia_model.pt
python convert_hf_to_openai.py --checkpoint models/Rithik101_WhispASR/ --whisper_dump_path openai/asr_model.pt
python convert_hf_to_openai.py --checkpoint models/steja_whisper-small-telugu-large-data/ --whisper_dump_path openai/telugu_model.pt
python convert_hf_to_openai.py --checkpoint models/vasista22_whisper-hindi-small/ --whisper_dump_path openai/hindi_model.pt
python convert_hf_to_openai.py --checkpoint models/vasista22_whisper-kannada-small/ --whisper_dump_path openai/kannada_model.pt

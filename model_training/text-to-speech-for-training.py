import os
import torch
from transformers import AutoTokenizer, AutoModelForTextToWaveform
import torchaudio

class TTSWrapper:
    def __init__(self, tokenizer, tts_model):
        self.tokenizer = tokenizer
        self.tts_model = tts_model

    def text_to_speech(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            speech = self.tts_model(**inputs)

        return speech

def main():
    # Initialize TTS model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    tts_model = AutoModelForTextToWaveform.from_pretrained("facebook/mms-tts-eng")

    tts_wrapper = TTSWrapper(tokenizer, tts_model)

    # Path to phrases.txt
    phrases_file = "phrases.txt"

    # Path to save generated speech
    output_dir = "generated_speech/"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read phrases from file and generate speech
    with open(phrases_file, 'r') as file:
        for idx, line in enumerate(file):
            phrase = line.strip()
            # Generate speech for the current phrase
            speech = tts_wrapper.text_to_speech(phrase)
            # Convert speech to a torch.Tensor
            speech_tensor = torch.tensor(speech[0], dtype=torch.float32)
            # Save the generated speech
            file_name = f"phrase_{idx}.wav"
            output_path = os.path.join(output_dir, file_name)
            try:
                torchaudio.save(output_path, speech_tensor, 22050)  # Adjust sample rate if needed
                print(f"Saved speech for phrase {idx}: {output_path}")
            except Exception as e:
                print(f"Error saving speech for phrase {idx}: {e}")

if __name__ == "__main__":
    main()

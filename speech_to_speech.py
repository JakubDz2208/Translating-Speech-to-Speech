from speechbrain.pretrained import EncoderClassifier
import torch
import pyaudio
import numpy as np
import wave
import sounddevice as sd
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForTextToWaveform
from IPython.display import Audio

class SpeechProcessor:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = "openai/whisper-large-v3"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, use_safetensors=False
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id, return_attention_mask=True)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=5,
            batch_size=16,
            return_timestamps=False,
            torch_dtype=self.torch_dtype,
            device=self.device,
            use_fast=False,
            generate_kwargs={"language": "english"},
        )
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        self.tts_model = AutoModelForTextToWaveform.from_pretrained("facebook/mms-tts-eng")
        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")

    def transcribe_audio_saved(self, filename):
        audio_data = np.fromfile(filename, dtype=np.int16)  # Load audio data from file
        transcription = self.pipe(audio_data)
        return transcription

    def record_audio(self, filename="test_recording", record_seconds=5, channels=1, rate=16000):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=2)
        print("Recording...")
        frames = []
        for i in range(0, int(rate / CHUNK * record_seconds)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("Finished recording.")
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

    def check_sound_device(self):
        print(sd.query_devices())

    def text_to_speech(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            speech = self.tts_model(**inputs).waveform

        return speech

    def classify_speaker(self, audio_file):
        waveform, sample_rate = torchaudio.load(audio_file)
        with torch.no_grad():
            embeddings = self.classifier.encode_batch(waveform)
            embeddings = torch.nn.functional.normalize(embeddings, dim=2)
            embeddings = embeddings.squeeze().cpu().numpy()
            embeddings = torch.tensor(embeddings).unsqueeze(0)
        return embeddings

    def transcribe_and_convert(self, audio_file):
        audio_rec = self.transcribe_audio_saved(audio_file)
        audio_text = audio_rec['text']
        speech = self.text_to_speech(audio_text)
        target_audio_file = f"translated_{audio_file}"
        return audio_text, speech
    

# speech_processor = SpeechProcessor()
# audio_file = input("\nEnter audio file name: ")
# audio_file = f"{audio_file}.wav"
# # speech_processor.record_audio(audio_file, record_seconds=10)
# transcription, speech_audio = speech_processor.transcribe_and_convert(audio_file)

# print("Transcription:", transcription)
# Audio(speech_audio.numpy(), rate=speech_processor.tts_model.config.sampling_rate)
# audio_data = speech_audio.numpy()
# sampling_rate = speech_processor.tts_model.config.sampling_rate

# import soundfile as sf
# sf.write("output_audio.wav", np.ravel(audio_data), sampling_rate)

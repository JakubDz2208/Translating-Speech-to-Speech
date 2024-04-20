import os
import sys
import pyaudio
import wave
from datetime import datetime

def generate_filename(output_dir, prefix, index):
    return os.path.join(output_dir, f"{prefix}_{index:03}.wav")

def generate_prefix():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")

def list_input_devices():
    audio = pyaudio.PyAudio()
    num_devices = audio.get_device_count()
    print("Available input devices:")
    for i in range(num_devices):
        info = audio.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0:
            print(f"{i}: {info['name']}")
    audio.terminate()

def record_audio(output_dir, prefix, phrases_file, input_device_index, duration=5, channels=1, sample_rate=44100, chunk_size=1024, format_=pyaudio.paInt16):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=format_,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size,
                        input_device_index=input_device_index)

    print("Recording...")

    try:
        with open(phrases_file, 'r') as file:
            for index, phrase in enumerate(file, start=1):
                filename = generate_filename(output_dir, prefix, index)
                phrase = phrase.strip()
                print(f"Please say: \"{phrase}\"")
                
                frames = []
                for i in range(0, int(sample_rate / chunk_size * duration)):
                    data = stream.read(chunk_size)
                    frames.append(data)

                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(audio.get_sample_size(format_))
                    wf.setframerate(sample_rate)
                    wf.writeframes(b''.join(frames))
                
                print(f"Recording {index} saved as {filename}.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    output_dir = "recordings"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    list_input_devices()
    
    input_device_index = int(input("Enter the index of the input device you want to use: "))
    
    prefix = generate_prefix()
    phrases_file = "phrases.txt"
    duration = 5
    
    try:
        record_audio(output_dir, prefix, phrases_file, input_device_index, duration=duration)
    except KeyboardInterrupt:
        print("\nRecording interrupted.")
        sys.exit(0)
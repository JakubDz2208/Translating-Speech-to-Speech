import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from speech_to_speech import SpeechProcessor
from IPython.display import Audio
import soundfile as sf
import numpy as np

class SpeechUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Processor UI")
        
        self.speech_processor = SpeechProcessor()

        self.label = tk.Label(root, text="Enter audio file name:")
        self.label.pack()

        self.entry = tk.Entry(root)
        self.entry.pack()

        self.record_button = tk.Button(root, text="Record Audio", command=self.record_audio)
        self.record_button.pack()

        self.transcribe_button = tk.Button(root, text="Transcribe and Convert", command=self.transcribe_and_convert)
        self.transcribe_button.pack()

    def record_audio(self):
        filename = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if filename:
            self.speech_processor.record_audio(filename, record_seconds=10)
            messagebox.showinfo("Recording Complete", "Audio recorded successfully.")

    def transcribe_and_convert(self):
        audio_file = self.entry.get()
        if not audio_file:
            messagebox.showerror("Error", "Please enter an audio file name.")
            return

        audio_file += ".wav"
        output_filename = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if not output_filename:
            messagebox.showwarning("Warning", "No output file chosen. Audio will not be saved.")
            return

        try:
            transcription, speech_audio = self.speech_processor.transcribe_and_convert(audio_file)
            messagebox.showinfo("Transcription", f"Transcription: {transcription}")
            Audio(speech_audio.numpy(), rate=self.speech_processor.tts_model.config.sampling_rate)

            # Save audio data to the chosen output file
            sf.write(output_filename, np.ravel(speech_audio.numpy()), self.speech_processor.tts_model.config.sampling_rate)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechUI(root)
    root.mainloop()

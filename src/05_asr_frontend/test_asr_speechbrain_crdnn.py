import tkinter as tk
from tkinter import scrolledtext
import threading
import io
import numpy as np
import soundfile as sf
import speech_recognition as sr
from transformers import pipeline

class RealTimeVoiceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Exoskeleton Voice AI")
        self.root.geometry("650x550")
        self.root.configure(bg="#1e1e1e")

        # --- UI ELEMENTS ---
        self.terminal = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, bg="#1e1e1e", fg="#00ffff", # Cyan text
            font=("Consolas", 11), state=tk.DISABLED, insertbackground="white"
        )
        self.terminal.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        input_frame = tk.Frame(root, bg="#1e1e1e")
        input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

        # The Toggle Button
        self.btn = tk.Button(
            input_frame, text="🎤 START CONTINUOUS LISTENING", bg="#4CAF50", fg="white", 
            font=("Consolas", 12, "bold"), command=self.toggle_listening
        )
        self.btn.pack(fill=tk.X, ipady=10)

        # --- MODEL & AUDIO INITIALIZATION ---
        self.recognizer = sr.Recognizer()
        self.is_listening = False

        self.print_to_terminal("[SYSTEM] Initializing Real-Time ASR Model...")
        self.print_to_terminal("         (Loading Distil-Whisper, please wait...)\n")
        self.root.update() 

        try:
            self.transcriber = pipeline(
                "automatic-speech-recognition", 
                model="distil-whisper/distil-small.en"
            )
            self.print_to_terminal("[SYSTEM] Voice AI loaded successfully!")
            self.print_to_terminal("Click the button below to activate hands-free mode.\n")
            self.print_to_terminal("-" * 60)
        except Exception as e:
            self.print_to_terminal(f"\n[ERROR] Failed to load ASR model: {e}")

    def print_to_terminal(self, text):
        self.terminal.config(state=tk.NORMAL)
        self.terminal.insert(tk.END, text + "\n")
        self.terminal.see(tk.END) 
        self.terminal.config(state=tk.DISABLED)

    # --- CONTINUOUS LISTENING LOGIC ---
    def toggle_listening(self):
        """Turns the microphone on or off"""
        if self.is_listening:
            # Turn OFF
            self.is_listening = False
            self.btn.config(text="🎤 START CONTINUOUS LISTENING", bg="#4CAF50")
            self.print_to_terminal("\n[SYSTEM] Microphone deactivated.")
        else:
            # Turn ON
            self.is_listening = True
            self.btn.config(text="🔴 STOP LISTENING", bg="#f44336")
            self.print_to_terminal("\n[SYSTEM] Continuous listening activated! Speak freely.")
            
            # Start the infinite listening loop in the background
            threading.Thread(target=self.continuous_record_loop, daemon=True).start()

    def continuous_record_loop(self):
        """Constantly listens to the mic, processes speech, and loops back"""
        with sr.Microphone() as source:
            # Calibrate to background noise once
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            while self.is_listening:
                try:
                    # Timeout=1 means it checks every 1 second if you clicked "Stop Listening"
                    # phrase_time_limit=5 prevents the AI from getting stuck if background noise is too loud
                    audio_data = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                    # If speech was detected, process it!
                    self.process_audio(audio_data)

                except sr.WaitTimeoutError:
                    # Normal behavior. Nobody spoke during that 1-second window. Just loop again.
                    continue
                except Exception as e:
                    if self.is_listening: # Only print errors if we didn't intentionally stop
                        self.print_to_terminal(f"  [ERROR] {e}")

    def process_audio(self, audio_data):
        """Converts raw audio bytes to a neural network array and transcribes"""
        self.print_to_terminal("\n> Processing speech...")
        self.root.update()

        try:
            # Convert Mic Data to AI format (In-Memory)
            wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
            wav_stream = io.BytesIO(wav_bytes)
            audio_array, sampling_rate = sf.read(wav_stream)
            audio_array = audio_array.astype(np.float32)

            # Run Inference
            result = self.transcriber({"sampling_rate": sampling_rate, "raw": audio_array})
            final_text = result["text"].strip().upper()

            if final_text:
                self.print_to_terminal(f"  AI Heard:  [{final_text}]")
            else:
                self.print_to_terminal("  [SYSTEM] Heard silence/noise.")

        except Exception as e:
            self.print_to_terminal(f"  [ERROR] {e}")

# ────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeVoiceGUI(root)
    root.mainloop()
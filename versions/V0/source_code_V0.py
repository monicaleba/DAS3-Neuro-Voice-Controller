import math
import threading
import time
import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import opensim as osim
import os
import sys
import io
import re
import soundfile as sf
import speech_recognition as sr
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ── Import the Schema ──
from command_schema import ALL_OUTPUTS, COMMANDS, TOKEN_OF

# --- Conda PATH Fix ---
conda_env_path = sys.prefix
conda_lib_bin = os.path.join(conda_env_path, "Library", "bin")
conda_std_bin = os.path.join(conda_env_path, "bin")
os.environ["PATH"] = conda_lib_bin + os.pathsep + conda_std_bin + os.pathsep + os.environ.get("PATH", "")
# ----------------------

MODEL_PATH = r"C:\Users\Lenovo\Desktop\OpenSIM Python\DAS3_release2\OpenSim_model\das3_ColoredAxes.osim"
TOKENIZER_DIR = './custom_das3_tokenizer'
SLM_MODEL_DIR = './custom_das3_Emilutz_SLM_AI_Model'

ARM_PATTERNS = ("SC_", "AC_", "GH_", "EL_", "PS_", "WR_")

# --- SAFETY GATEKEEPER DICTIONARY ---
PRIORITY_WORDS = {
    'SHOULDER_UP': ['shoulder', 'up', 'upper arm', 'upper limb', 'arm', 'upward'],
    'SHOULDER_DOWN': ['shoulder', 'down', 'upper arm', 'upper limb', 'arm', 'downward'],
    'SHOULDER_LEFT': ['shoulder', 'left', 'upper arm', 'upper limb', 'arm', 'leftward'],
    'SHOULDER_RIGHT': ['shoulder', 'right', 'upper arm', 'upper limb', 'arm', 'rightward'],
    'ELBOW_UP': ['elbow', 'up', 'forearm', 'lower arm', 'lower limb', 'in', 'upward'],
    'ELBOW_DOWN': ['elbow', 'down', 'forearm', 'lower arm', 'lower limb', 'out', 'downward'],
    'ROTATE_WRIST': ['spin', 'twist', 'roll', 'swivel', 'gyrate', 'wrist', 'hand'],
    'STOP': ['halt', 'freeze', 'abort', 'cancel', 'cease', 'stop', 'do not', "don't", 'end', 'break', 'quit', 'pause'],
    'REST': ['zero', 'home', 'sleep', 'baseline', 'dormant', 'rest', 'relax', 'neutral', 'default', 'calm', 'steady', 'nap'],
}

def load_model(model_path: str = MODEL_PATH):
    model = osim.Model(model_path)
    model.setUseVisualizer(True)
    state = model.initSystem()
    try:
        viz = model.updVisualizer().updSimbodyVisualizer()
        viz.setBackgroundType(viz.SolidColor)
        viz.setBackgroundColor(osim.Vec3(1.0, 1.0, 1.0))
    except Exception: pass
    return model, state

def get_arm_coords(model, state):
    cs = model.getCoordinateSet()
    coords = []
    for k in range(cs.getSize()):
        c  = cs.get(k)
        nm = c.getName()
        if not any(nm.startswith(p) for p in ARM_PATTERNS): continue
        if c.getLocked(state): c.setLocked(state, False)
        coords.append({
            "name": nm,
            "min_deg": math.degrees(c.getRangeMin()),
            "max_deg": math.degrees(c.getRangeMax()),
            "idx": k,
        })
    return coords

class SuperExoControllerGUI:
    FPS = 30
    DT = 1.0 / FPS
    SPEED = 40.0 # deg/s

    def __init__(self):
        self._build_ui()
        
        # OpenSim Kinematics
        self.model, self.state = load_model()
        self.coords = get_arm_coords(self.model, self.state)
        self.n = len(self.coords)
        self.coord_map = {c["name"]: i for i, c in enumerate(self.coords)}

        self.q = np.zeros(self.n)
        self.q_tgt = np.zeros(self.n)
        
        self.running = True
        self.paused = False
        self.speed = self.SPEED 
        
        # Audio & AI State
        self.recognizer = sr.Recognizer()
        self.is_listening = False

        self._set_resting_pose()
        
        # Threading & Loading
        threading.Thread(target=self._load_ai_models, daemon=True).start()
        self._start_animation_thread()

    # ─── KINEMATICS LOGIC ─────────────────────────────────────────────────
    def _set_target(self, joint_name, angle):
        if joint_name in self.coord_map:
            idx = self.coord_map[joint_name]
            min_a = self.coords[idx]["min_deg"]
            max_a = self.coords[idx]["max_deg"]
            self.q_tgt[idx] = np.clip(angle, min_a, max_a)
            
    def _adjust_target(self, joint_name, delta_angle):
        if joint_name in self.coord_map:
            idx = self.coord_map[joint_name]
            self._set_target(joint_name, self.q_tgt[idx] + delta_angle)

    def _set_resting_pose(self):
        self._set_zero_pose()
        self._set_target("GH_z", 10) # Slight natural resting drop

    def _set_zero_pose(self):
        for coord in self.coords:
            self._set_target(coord["name"], 0)

    # ─── AI LOADING ───────────────────────────────────────────────────────
    def _load_ai_models(self):
        self.print_to_terminal("[SYSTEM] Initializing Master AI Pipeline...")
        try:
            # 1. Load Voice AI (Distil-Whisper)
            self.print_to_terminal("[SYSTEM] Loading Layer 1: Voice Recognition (Whisper)...")
            self.asr_pipeline = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en")
            
            # 2. Load SLM Brain
            self.print_to_terminal("[SYSTEM] Loading Layer 2: NLP Brain (Custom SLM)...")
            self.slm_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
            self.slm_model = AutoModelForSequenceClassification.from_pretrained(SLM_MODEL_DIR)
            self.slm_model.eval()
            self.id2label = self.slm_model.config.id2label
            
            self.print_to_terminal("[SYSTEM] AI SYSTEM ONLINE. Ready for commands.\n" + "-"*50)
            self.root.after(0, lambda: self.lbl_status.config(text="SYSTEM ONLINE", fg="#4CAF50"))
        except Exception as e:
            self.print_to_terminal(f"\n[CRITICAL ERROR] Failed to load models: {e}")
            self.root.after(0, lambda: self.lbl_status.config(text="AI FAILURE", fg="#f44336"))

    # ─── MASTER NLP PIPELINE & GATEKEEPER ─────────────────────────────────
    def process_input_text(self, text):
        """Passes text through the Gatekeeper, then the SLM, then dispatches to OpenSim"""
        text_lower = text.lower()
        has_stop_word = False
        other_words_found = set()

        # 1. The Gatekeeper Rule (1 for Stop, 2 for everything else)
        for category, words in PRIORITY_WORDS.items():
            for word in words:
                if re.search(r'\b' + re.escape(word.lower()) + r'\b', text_lower):
                    if category == 'STOP': has_stop_word = True
                    else: other_words_found.add(word.lower())

        if has_stop_word or len(other_words_found) >= 2:
            pass # Gatekeeper allows it
        else:
            self.print_to_terminal("  [GATEKEEPER] BLOCKED (Not enough valid keywords)")
            return "UNKNOWN"

        # 2. SLM Inference
        inputs = self.slm_tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            outputs = self.slm_model(**inputs)
            logits = outputs.logits
            prob = torch.max(F.softmax(logits, dim=1)).item()
            predicted_id = torch.argmax(logits, dim=1).item()
            
            raw_label = self.id2label[str(predicted_id)] if str(predicted_id) in self.id2label else self.id2label[predicted_id]

        if prob < 0.60:
            self.print_to_terminal(f"  [SLM] BLOCKED (Low Confidence: {prob:.2f})")
            return "UNKNOWN"

        # 3. Translation to Standard Token
        token = TOKEN_OF.get(raw_label, raw_label)
        self.print_to_terminal(f"  [SLM DECISION] {token} (Conf: {prob:.2f})")
        return token

    def dispatch_command(self, token):
        """Translates the NLP token to physical OpenSim arm movements"""
        if token == "UNKNOWN":
            self.root.after(0, lambda: self.lbl_command.config(text="Action: REJECTED", fg="#f44336"))
            return

        self.root.after(0, lambda: self.lbl_command.config(text=f"Action: {token}", fg="#00ffff"))

        # Physical Exoskeleton Execution Layer
        if token == 'STOP':
            self.q_tgt = np.copy(self.q) # Halt instantly at current angles
        elif token == 'REST':
            self._set_resting_pose()
        elif token == 'SHOULDER_UP':
            self._adjust_target("GH_z", 20)
        elif token == 'SHOULDER_DOWN':
            self._adjust_target("GH_z", -20)
        elif token == 'SHOULDER_LEFT':
            self._adjust_target("GH_y", 20)
        elif token == 'SHOULDER_RIGHT':
            self._adjust_target("GH_y", -20)
        elif token == 'ELBOW_UP':
            self._adjust_target("EL_x", 20)
        elif token == 'ELBOW_DOWN':
            self._adjust_target("EL_x", -20)
        elif token == 'ROTATE_WRIST':
            self._adjust_target("PS_y", 45)

    # ─── TEXT INPUT HANDLER ───────────────────────────────────────────────
    def _on_text_enter(self, event=None):
        user_input = self.entry_box.get().strip()
        if not user_input: return
        self.entry_box.delete(0, tk.END)
        
        self.print_to_terminal(f"\n> [TEXT IN]: '{user_input}'")
        token = self.process_input_text(user_input)
        self.dispatch_command(token)

    # ─── VOICE LISTENING LOGIC ────────────────────────────────────────────
    def toggle_listening(self):
        if self.is_listening:
            self.is_listening = False
            self.btn_mic.config(text="🎤 START CONTINUOUS LISTENING", bg="#4CAF50")
            self.print_to_terminal("\n[SYSTEM] Microphone deactivated.")
        else:
            self.is_listening = True
            self.btn_mic.config(text="🔴 STOP LISTENING", bg="#f44336")
            self.print_to_terminal("\n[SYSTEM] Continuous listening activated! Speak freely.")
            threading.Thread(target=self.continuous_record_loop, daemon=True).start()

    def continuous_record_loop(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            while self.is_listening:
                try:
                    audio_data = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    self._process_voice(audio_data)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    if self.is_listening: self.print_to_terminal(f"  [ERROR] {e}")

    def _process_voice(self, audio_data):
        self.print_to_terminal("\n> Processing speech...")
        self.root.update()
        try:
            # Convert Mic Data to Whisper Array
            wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
            wav_stream = io.BytesIO(wav_bytes)
            audio_array, sr_rate = sf.read(wav_stream)
            audio_array = audio_array.astype(np.float32)

            # Whisper Inference
            result = self.asr_pipeline({"sampling_rate": sr_rate, "raw": audio_array})
            final_text = result["text"].strip().upper()

            if final_text:
                self.print_to_terminal(f"  [MIC IN]: '{final_text}'")
                token = self.process_input_text(final_text)
                self.dispatch_command(token)
            else:
                self.print_to_terminal("  [SYSTEM] Heard silence.")
        except Exception as e:
            self.print_to_terminal(f"  [ERROR] {e}")

    # ─── ANIMATION THREAD (OPENSIM) ───────────────────────────────────────
    def _animation_loop(self):
        while self.running:
            t0 = time.perf_counter()
            if not self.paused:
                err = self.q_tgt - self.q
                step = np.sign(err) * np.minimum(np.abs(err), self.speed * self.DT)
                self.q += step

            cs = self.model.getCoordinateSet()
            for k in range(self.n):
                try:
                    c = cs.get(self.coords[k]["idx"])
                    c.setValue(self.state, math.radians(self.q[k]))
                except Exception: pass

            try:
                self.model.realizePosition(self.state)
                self.model.getVisualizer().show(self.state)
            except Exception: pass

            elapsed = time.perf_counter() - t0
            time.sleep(max(0, self.DT - elapsed))

    def _start_animation_thread(self):
        threading.Thread(target=self._animation_loop, daemon=True).start()

    # ─── UI BUILDING ──────────────────────────────────────────────────────
    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("DAS3 Neuro-Voice & Text Controller")
        self.root.geometry("650x700")
        self.root.configure(bg="#0F1923")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 1. Headers
        tk.Label(self.root, text="Exoskeleton Command Dashboard", bg="#1E5C8A", fg="white", font=("Arial", 14, "bold")).pack(fill=tk.X, ipady=8)
        self.lbl_status = tk.Label(self.root, text="Booting...", bg="#0F1923", fg="#F5C842", font=("Arial", 11, "bold"))
        self.lbl_status.pack(pady=5)
        self.lbl_command = tk.Label(self.root, text="Action: WAITING", bg="#0F1923", fg="#FFD43B", font=("Arial", 16, "bold"))
        self.lbl_command.pack(pady=5)

        # 2. Terminal
        self.terminal = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, bg="#111D2A", fg="#A3C2C2", font=("Consolas", 10), state=tk.DISABLED, insertbackground="white")
        self.terminal.pack(padx=15, pady=5, fill=tk.BOTH, expand=True)

        # 3. Text Input Frame
        input_frame = tk.Frame(self.root, bg="#0F1923")
        input_frame.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(input_frame, text="Manual Entry >", bg="#0F1923", fg="white", font=("Consolas", 11)).pack(side=tk.LEFT)
        self.entry_box = tk.Entry(input_frame, font=("Consolas", 12), bg="#1e293b", fg="white", insertbackground="white")
        self.entry_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, ipady=5)
        self.entry_box.bind("<Return>", self._on_text_enter)
        
        self.btn_send = tk.Button(input_frame, text="SEND", command=self._on_text_enter, bg="#2A5C8A", fg="white", font=("Arial", 10, "bold"), relief=tk.FLAT)
        self.btn_send.pack(side=tk.RIGHT, ipady=3, ipadx=10)

        # 4. Voice Toggle Button
        self.btn_mic = tk.Button(self.root, text="🎤 START CONTINUOUS LISTENING", bg="#4CAF50", fg="white", font=("Consolas", 14, "bold"), command=self.toggle_listening)
        self.btn_mic.pack(fill=tk.X, padx=15, pady=(0, 15), ipady=12)

    def print_to_terminal(self, text):
        self.terminal.config(state=tk.NORMAL)
        self.terminal.insert(tk.END, text + "\n")
        self.terminal.see(tk.END) 
        self.terminal.config(state=tk.DISABLED)

    def _on_close(self):
        self.running = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = SuperExoControllerGUI()
    app.run()
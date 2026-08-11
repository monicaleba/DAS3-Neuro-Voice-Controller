import argparse
import math
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import opensim as osim
import os
import sys

# --- THE FIX: Point OpenSim to Conda's hidden bin folder ---
conda_env_path = sys.prefix
conda_lib_bin = os.path.join(conda_env_path, "Library", "bin")
conda_std_bin = os.path.join(conda_env_path, "bin")

# Add both possible Conda bin locations to the Windows PATH
os.environ["PATH"] = conda_lib_bin + os.pathsep + conda_std_bin + os.pathsep + os.environ.get("PATH", "")
# -----------------------------------------------------------

# *** IMPORTANT: Point this to your PATCHED model file ***
MODEL_PATH = r"C:\Users\Lenovo\Desktop\OpenSIM Python\DAS3_release2\OpenSim_model\das3_wrist_unlocked.osim"

def load_model(model_path: str = MODEL_PATH, use_visualizer: bool = True):
    """Load DAS3 and return (model, state)."""
    model = osim.Model(model_path)
    
    # --- VISUAL AXES: Inject (Red=X, Green=Y, Blue=Z) ---
    target_keywords = ["sternoclavicular", "acromioclavicular", "glenohumeral",
                       "shoulder", "elbow", "radioulnar", "rc"]
    for i in range(model.getJointSet().getSize()):
        joint = model.getJointSet().get(i)
        joint_name = joint.getName().lower()
        if any(keyword in joint_name for keyword in target_keywords):
            axes = osim.FrameGeometry()
            axes.set_scale_factors(osim.Vec3(0.12))
            joint.updChildFrame().attachGeometry(axes)
    # ----------------------------------------------------
    
    if use_visualizer:
        model.setUseVisualizer(True)
    state = model.initSystem()
    return model, state

# ── Right-arm coordinate patterns (WR_ added for wrist) ─────────────
ARM_PATTERNS = ("GH_", "EL_", "PS_", "WR_", "SC_", "AC_")

def get_arm_coords(model, state):
    """Return list of right-arm coordinate dicts (FORCED UNLOCKED)."""
    cs = model.getCoordinateSet()
    coords = []
    for k in range(cs.getSize()):
        c  = cs.get(k)
        nm = c.getName()
        
        # 1. Skip if it's not part of the arm
        if not any(nm.startswith(p) for p in ARM_PATTERNS):
            continue
        
        # 2. Force unlock the coordinate if it was locked
        if c.getLocked(state):
            c.setLocked(state, False)
            
        # 3. Add it to our slider list
        coords.append({
            "name":    nm,
            "label":   nm.replace("_", " "),
            "min_deg": math.degrees(c.getRangeMin()),
            "max_deg": math.degrees(c.getRangeMax()),
            "idx":     k,
        })
    return coords

class DAS3SliderApp:
    """Main application window."""

    FPS     = 30
    DT      = 1.0 / FPS
    SPEED   = 30.0  # deg/s default interpolation speed

    def __init__(self, model_path: str = MODEL_PATH):
        self.model, self.state = load_model(model_path, use_visualizer=True)
        self.coords = get_arm_coords(self.model, self.state)
        self.n      = len(self.coords)

        self.q     = np.zeros(self.n)
        self.q_tgt = np.zeros(self.n)
        self.speed = self.SPEED
        self.running = True

        self._build_ui()
        self._start_animation_thread()

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("DAS3_release2 — Slider Control (Wrist Unlocked)")
        self.root.configure(bg="#0F1923")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        LEFT_W = 720
        LEFT_H = 620

        # ── 3D mirror canvas
        self.canvas = tk.Canvas(self.root, width=LEFT_W, height=LEFT_H,
                                bg="#0A1018", highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=2, padx=12, pady=12, sticky="nsew")
        self._draw_torso()

        # ── Scrollable Slider Panel
        slider_outer = tk.Frame(self.root, bg="#111D2A", bd=1, relief=tk.RIDGE)
        slider_outer.grid(row=0, column=1, padx=(0,12), pady=(12, 0), sticky="nsew")

        slider_canvas = tk.Canvas(slider_outer, bg="#111D2A",
                                  highlightthickness=0, width=540)
        scrollbar = ttk.Scrollbar(slider_outer, orient="vertical",
                                  command=slider_canvas.yview)
        slider_canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        slider_canvas.pack(side="left", fill="both", expand=True)

        PANEL = tk.Frame(slider_canvas, bg="#111D2A")
        canvas_window = slider_canvas.create_window((0, 0), window=PANEL, anchor="nw")

        def on_frame_configure(event):
            slider_canvas.configure(scrollregion=slider_canvas.bbox("all"))
        PANEL.bind("<Configure>", on_frame_configure)

        def on_canvas_configure(event):
            slider_canvas.itemconfig(canvas_window, width=event.width)
        slider_canvas.bind("<Configure>", on_canvas_configure)

        def on_mousewheel(event):
            slider_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        slider_canvas.bind_all("<MouseWheel>", on_mousewheel)

        tk.Label(PANEL, text=" Joint Control — Sliders (Wrist Unlocked) ",
                 bg="#1E5C8A", fg="white",
                 font=("Arial", 11, "bold")).pack(fill=tk.X, ipady=4)

        COLORS = ["#F28020","#4ABDE8","#52DC84","#D864D8","#F5C842","#52E8E0"]
        self.sl_vars = []
        self.val_vars = []

        for j, coord in enumerate(self.coords):
            color = COLORS[j % len(COLORS)]
            row = tk.Frame(PANEL, bg="#111D2A")
            row.pack(fill=tk.X, padx=10, pady=4)
            tk.Label(row,
                     text=f"{coord['label']} [{coord['name']}]",
                     fg=color, bg="#111D2A",
                     font=("Arial", 9, "bold")).pack(anchor="w")

            sl_var = tk.DoubleVar(value=0.0)
            self.sl_vars.append(sl_var)

            sl_row = tk.Frame(row, bg="#111D2A")
            sl_row.pack(fill=tk.X)
            sl = ttk.Scale(sl_row,
                           from_=coord["min_deg"], to=coord["max_deg"],
                           variable=sl_var, orient=tk.HORIZONTAL, length=380)
            sl.configure(command=lambda v, idx=j: self._on_slider(idx, float(v)))
            sl.pack(side=tk.LEFT)

            val_var = tk.StringVar(value="  0°")
            self.val_vars.append(val_var)
            tk.Label(sl_row, textvariable=val_var, width=7,
                     fg="#FFD43B", bg="#111D2A",
                     font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=6)

            rom_txt = f"[{coord['min_deg']:+.0f}° … {coord['max_deg']:+.0f}°]"
            tk.Label(row, text=rom_txt, fg="#5A7A8A", bg="#111D2A",
                     font=("Arial", 8)).pack(anchor="w")

        # ── Speed slider
        sp_frame = tk.Frame(PANEL, bg="#111D2A")
        sp_frame.pack(fill=tk.X, padx=10, pady=6)
        tk.Label(sp_frame, text="Movement speed",
                 fg="#8AA8C0", bg="#111D2A",
                 font=("Arial",9,"bold")).pack(anchor="w")
        sp_row = tk.Frame(sp_frame, bg="#111D2A")
        sp_row.pack(fill=tk.X)
        self.speed_var = tk.DoubleVar(value=self.SPEED)
        ttk.Scale(sp_row, from_=2, to=120,
                  variable=self.speed_var, orient=tk.HORIZONTAL, length=380,
                  command=lambda v: self._on_speed(float(v))).pack(side=tk.LEFT)
        self.speed_label = tk.Label(sp_row, text="30 °/s", width=7,
                                    fg="#4CEA80", bg="#111D2A",
                                    font=("Arial",9,"bold"))
        self.speed_label.pack(side=tk.LEFT, padx=6)

        # ── Buttons
        btn_frame = tk.Frame(PANEL, bg="#111D2A")
        btn_frame.pack(fill=tk.X, padx=10, pady=8)
        tk.Button(btn_frame, text="Reset all joints  (R)",
                  command=self._reset_all,
                  bg="#2A5C8A", fg="white",
                  font=("Arial",10,"bold"),
                  relief=tk.FLAT,
                  activebackground="#4472C4").pack(fill=tk.X, ipady=4)

        # ── Readouts (show up to 7 joints now including wrist)
        ro_frame = tk.LabelFrame(self.root,
                                 text=" Joint Angles & EE Position ",
                                 fg="#4ABDE8", bg="#0F1923",
                                 font=("Arial",9,"bold"))
        ro_frame.grid(row=1, column=1, padx=(0,12), pady=(12,12), sticky="ew")
        self.ro_vars = []
        display_count = min(len(self.coords), 7)
        for j in range(display_count):
            coord = self.coords[j]
            ro_var = tk.StringVar(value="+0.0°")
            self.ro_vars.append(ro_var)
            tk.Label(ro_frame, text=f"{coord['label']}:",
                     fg="#8898A8", bg="#0F1923", width=22,
                     anchor="w").grid(row=j, column=0, padx=8, pady=1)
            tk.Label(ro_frame, textvariable=ro_var, width=8,
                     fg="#FFD43B", bg="#0F1923", anchor="e",
                     font=("Arial",9,"bold")).grid(row=j, column=1, padx=4)

        self.root.bind("<r>", lambda e: self._reset_all())
        self.root.bind("<R>", lambda e: self._reset_all())

    def _on_slider(self, idx: int, val: float) -> None:
        self.q_tgt[idx] = val

    def _on_speed(self, val: float) -> None:
        self.speed = val
        self.speed_label.configure(text=f"{val:.0f} °/s")

    def _reset_all(self) -> None:
        self.q_tgt[:] = 0.0
        for var in self.sl_vars:
            var.set(0.0)

    def _on_close(self) -> None:
        self.running = False
        self.root.destroy()

    def _start_animation_thread(self) -> None:
        t = threading.Thread(target=self._animation_loop, daemon=True)
        t.start()

    def _animation_loop(self) -> None:
        while self.running:
            t0 = time.perf_counter()
            err  = self.q_tgt - self.q
            step = np.sign(err) * np.minimum(np.abs(err), self.speed * self.DT)
            self.q += step

            cs = self.model.getCoordinateSet()
            for k in range(self.n):
                try:
                    c = cs.get(self.coords[k]["idx"])
                    c.setValue(self.state, math.radians(self.q[k]))
                except Exception:
                    pass
            try:
                self.model.realizePosition(self.state)
                self.model.getVisualizer().show(self.state)
            except Exception:
                pass

            self.root.after(0, self._update_ui)
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, self.DT - elapsed))

    def _update_ui(self) -> None:
        for j in range(self.n):
            self.val_vars[j].set(f"{self.q[j]:+.0f}°")
            if j < len(self.ro_vars):
                self.ro_vars[j].set(f"{self.q[j]:+.1f}°")

    def _draw_torso(self) -> None:
        CX, CY = 280, 310
        self.canvas.create_rectangle(CX-60, CY-160, CX+60, CY+60,
                                     fill="#1C3A5A", outline="#2A6090", width=1.5)
        self.canvas.create_oval(CX-50, CY-240, CX+50, CY-160,
                                fill="#7A5C4A", outline="#9A7A62", width=1.5)
        self.canvas.create_oval(CX+55,CY-135,CX+75,CY-115,
                                fill="#FFD43B",outline="white",width=1)

    def run(self) -> None:
        self.root.mainloop()

if __name__ == "__main__":
    print("Launching DAS3 Slider Controller (Wrist Unlocked)...")
    print(f"Model: {MODEL_PATH}")
    app = DAS3SliderApp()
    app.run()
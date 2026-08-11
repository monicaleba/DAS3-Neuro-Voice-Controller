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

# --- Conda PATH Fix ---
conda_env_path = sys.prefix
conda_lib_bin = os.path.join(conda_env_path, "Library", "bin")
conda_std_bin = os.path.join(conda_env_path, "bin")
os.environ["PATH"] = conda_lib_bin + os.pathsep + conda_std_bin + os.pathsep + os.environ.get("PATH", "")
# ----------------------

# LOAD THE NEW MODIFIED MODEL!
MODEL_PATH = r"C:\Users\Lenovo\Desktop\OpenSIM Python\DAS3_release2\OpenSim_model\das3_ColoredAxes.osim"

JOINT_COLORS = {
    "sc1": (1.0, 0.2, 0.2),    # SC_y 
    "sc2": (1.0, 0.5, 0.0),    # SC_z 
    "sc3": (1.0, 0.85, 0.0),   # SC_x 
    "ac1": (0.0, 0.8, 0.2),    # AC_y 
    "ac2": (0.0, 1.0, 0.6),    # AC_z 
    "ac3": (0.2, 0.9, 0.9),    # AC_x 
    "gh1": (0.2, 0.4, 1.0),    # GH_y 
    "gh2": (0.5, 0.3, 1.0),    # GH_z 
    "gh3": (0.7, 0.2, 1.0),    # GH_yy
    "hu":  (1.0, 0.0, 0.5),    # EL_x 
    "ur":  (0.9, 0.3, 0.6),    # PS_y 
    "rc":  (0.6, 0.4, 0.2),    # WR_x, WR_z 
}

COORD_TO_JOINT = {
    "SC_y": "sc1", "SC_z": "sc2", "SC_x": "sc3",
    "AC_y": "ac1", "AC_z": "ac2", "AC_x": "ac3",
    "GH_y": "gh1", "GH_z": "gh2", "GH_yy": "gh3",
    "EL_x": "hu",  "PS_y": "ur",
    "WR_x": "rc",  "WR_z": "rc",
}

def load_model(model_path: str = MODEL_PATH, use_visualizer: bool = True):
    model = osim.Model(model_path)
    if use_visualizer:
        model.setUseVisualizer(True)

    state = model.initSystem()

    # Apply White Background
    if use_visualizer:
        try:
            viz = model.updVisualizer().updSimbodyVisualizer()
            viz.setBackgroundType(viz.SolidColor)
            viz.setBackgroundColor(osim.Vec3(1.0, 1.0, 1.0))
        except Exception as e:
            print(f" [WARN] Could not set white background: {e}")

    return model, state

ARM_PATTERNS = ("GH_", "EL_", "PS_", "WR_", "SC_", "AC_")

def get_arm_coords(model, state):
    cs = model.getCoordinateSet()
    coords = []
    for k in range(cs.getSize()):
        c  = cs.get(k)
        nm = c.getName()
        if not any(nm.startswith(p) for p in ARM_PATTERNS): continue
        if c.getLocked(state):
            c.setLocked(state, False)
        coords.append({
            "name":    nm,
            "label":   nm.replace("_", " "),
            "min_deg": math.degrees(c.getRangeMin()),
            "max_deg": math.degrees(c.getRangeMax()),
            "idx":     k,
        })
    return coords

def rgb_to_hex(r, g, b):
    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"

class DAS3SliderApp:
    FPS     = 30
    DT      = 1.0 / FPS
    SPEED   = 30.0

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
        self.root.title("DAS3_release2 — Slider Control (Color-Coded Joints)")
        self.root.configure(bg="#0F1923")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)

        # ── Scrollable Slider Panel
        slider_outer = tk.Frame(self.root, bg="#111D2A", bd=1, relief=tk.RIDGE)
        slider_outer.grid(row=0, column=0, padx=12, pady=(12, 0), sticky="nsew")

        slider_canvas = tk.Canvas(slider_outer, bg="#111D2A", highlightthickness=0)
        scrollbar = ttk.Scrollbar(slider_outer, orient="vertical", command=slider_canvas.yview)
        slider_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        slider_canvas.pack(side="left", fill="both", expand=True)

        PANEL = tk.Frame(slider_canvas, bg="#111D2A")
        canvas_window = slider_canvas.create_window((0, 0), window=PANEL, anchor="nw")
        PANEL.columnconfigure(0, weight=1)
        PANEL.columnconfigure(1, weight=1)

        def on_frame_configure(event): slider_canvas.configure(scrollregion=slider_canvas.bbox("all"))
        PANEL.bind("<Configure>", on_frame_configure)
        def on_canvas_configure(event): slider_canvas.itemconfig(canvas_window, width=event.width)
        slider_canvas.bind("<Configure>", on_canvas_configure)
        def on_mousewheel(event): slider_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        slider_canvas.bind_all("<MouseWheel>", on_mousewheel)

        tk.Label(PANEL, text=" Joint Control — Sliders ", bg="#1E5C8A", fg="white", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="ew", ipady=4, pady=(0, 10))

        self.sl_vars = []
        self.val_vars = []

        for j, coord in enumerate(self.coords):
            joint_key = COORD_TO_JOINT.get(coord["name"], None)
            if joint_key and joint_key in JOINT_COLORS:
                r, g, b = JOINT_COLORS[joint_key]
                color = rgb_to_hex(r, g, b)
            else: color = "#AAAAAA"

            row_idx, col_idx = (j // 2) + 1, j % 2
            cell = tk.Frame(PANEL, bg="#111D2A")
            cell.grid(row=row_idx, column=col_idx, sticky="ew", padx=15, pady=6)

            tk.Label(cell, text=f"● {coord['label']} [{coord['name']}]", fg=color, bg="#111D2A", font=("Arial", 9, "bold")).pack(anchor="w")

            sl_var = tk.DoubleVar(value=0.0)
            self.sl_vars.append(sl_var)
            sl_row = tk.Frame(cell, bg="#111D2A")
            sl_row.pack(fill=tk.X)

            sl = ttk.Scale(sl_row, from_=coord["min_deg"], to=coord["max_deg"], variable=sl_var, orient=tk.HORIZONTAL)
            sl.configure(command=lambda v, idx=j: self._on_slider(idx, float(v)))
            sl.pack(side=tk.LEFT, fill=tk.X, expand=True)

            val_var = tk.StringVar(value="  0°")
            self.val_vars.append(val_var)
            tk.Label(sl_row, textvariable=val_var, width=7, fg="#FFD43B", bg="#111D2A", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=6)
            tk.Label(cell, text=f"[{coord['min_deg']:+.0f}° … {coord['max_deg']:+.0f}°]", fg="#5A7A8A", bg="#111D2A", font=("Arial", 8)).pack(anchor="w")

        bottom_row = (len(self.coords) // 2) + 2

        # ── Speed slider & Reset
        sp_frame = tk.Frame(PANEL, bg="#111D2A")
        sp_frame.grid(row=bottom_row, column=0, columnspan=2, sticky="ew", padx=15, pady=(20, 6))
        tk.Label(sp_frame, text="Movement speed", fg="#8AA8C0", bg="#111D2A", font=("Arial", 9, "bold")).pack(anchor="w")
        sp_row = tk.Frame(sp_frame, bg="#111D2A")
        sp_row.pack(fill=tk.X)
        self.speed_var = tk.DoubleVar(value=self.SPEED)
        sp_scale = ttk.Scale(sp_row, from_=2, to=120, variable=self.speed_var, orient=tk.HORIZONTAL)
        sp_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.speed_label = tk.Label(sp_row, text="30 °/s", width=7, fg="#4CEA80", bg="#111D2A", font=("Arial", 9, "bold"))
        self.speed_label.pack(side=tk.LEFT, padx=6)
        sp_scale.configure(command=lambda v: self._on_speed(float(v)))

        btn_frame = tk.Frame(PANEL, bg="#111D2A")
        btn_frame.grid(row=bottom_row+1, column=0, columnspan=2, sticky="ew", padx=15, pady=(5, 15))
        tk.Button(btn_frame, text="Reset all joints  (R)", command=self._reset_all, bg="#2A5C8A", fg="white", font=("Arial", 10, "bold"), relief=tk.FLAT, activebackground="#4472C4").pack(fill=tk.X, ipady=4)

        # ── LEGEND PANEL (Main Line Colors) ──────────────────────────────────
        legend_outer = tk.LabelFrame(self.root, text=" ◆ Joint Model Legend ", fg="#4ABDE8", bg="#0F1923", font=("Arial", 10, "bold"), bd=2, relief=tk.GROOVE)
        legend_outer.grid(row=0, column=1, padx=(0, 12), pady=(12, 0), sticky="ns")

        legend_groups = [
            ("STERNOCLAVICULAR (SC)", [("sc1", "SC_y — Protraction / Retraction"), ("sc2", "SC_z — Elevation / Depression"), ("sc3", "SC_x — Axial Rotation")]),
            ("ACROMIOCLAVICULAR (AC)", [("ac1", "AC_y — Protraction / Retraction"), ("ac2", "AC_z — Lateral Tilt"), ("ac3", "AC_x — Forward Tilt")]),
            ("GLENOHUMERAL (GH)", [("gh1", "GH_y — Plane of Elevation"), ("gh2", "GH_z — Elevation Angle"), ("gh3", "GH_yy — Axial Rotation")]),
            ("ELBOW (EL)", [("hu", "EL_x — Flexion / Extension")]),
            ("PRO/SUPINATION (PS)", [("ur", "PS_y — Pronation / Supination")]),
            ("WRIST (WR)", [("rc", "WR_x,z — Flex & Deviation")]),
        ]

        for group_title, items in legend_groups:
            tk.Label(legend_outer, text=group_title, fg="#8AA8C0", bg="#0F1923", font=("Arial", 8, "bold"), anchor="w").pack(fill=tk.X, padx=8, pady=(8, 2))
            for joint_key, description in items:
                r, g, b = JOINT_COLORS[joint_key]
                hex_color = rgb_to_hex(r, g, b)
                item_frame = tk.Frame(legend_outer, bg="#0F1923")
                item_frame.pack(fill=tk.X, padx=12, pady=1)
                swatch = tk.Canvas(item_frame, width=14, height=14, bg=hex_color, highlightthickness=1, highlightbackground="#333333")
                swatch.pack(side=tk.LEFT, padx=(0, 6))
                tk.Label(item_frame, text=description, fg=hex_color, bg="#0F1923", font=("Arial", 8), anchor="w").pack(side=tk.LEFT)

        # ── Axis Color Legend (Added to the bottom of the right panel) ──
        # Subtle separator line
        tk.Frame(legend_outer, bg="#2A3B4C", height=1).pack(fill=tk.X, padx=10, pady=(15, 5))
        
        tk.Label(legend_outer, text="TIP CUBES (AXIS COLORS)", fg="#8AA8C0", bg="#0F1923", font=("Arial", 8, "bold"), anchor="w").pack(fill=tk.X, padx=8, pady=(4, 2))
        
        axis_colors = [("X-Axis (Red)", "#FF0000"), ("Y-Axis (Green)", "#00CC00"), ("Z-Axis (Blue)", "#0000FF")]
        for ax_name, hex_c in axis_colors:
            item_f = tk.Frame(legend_outer, bg="#0F1923")
            item_f.pack(fill=tk.X, padx=12, pady=1)
            c = tk.Canvas(item_f, width=14, height=14, bg=hex_c, highlightthickness=1, highlightbackground="#333333")
            c.pack(side=tk.LEFT, padx=(0,6))
            tk.Label(item_f, text=ax_name, fg=hex_c, bg="#0F1923", font=("Arial", 8, "bold"), anchor="w").pack(side=tk.LEFT)

        # ── Readouts
        ro_frame = tk.LabelFrame(self.root, text=" Joint Angles ", fg="#4ABDE8", bg="#0F1923", font=("Arial", 9, "bold"))
        ro_frame.grid(row=1, column=0, columnspan=2, padx=12, pady=12, sticky="ew")
        for i in range(min(7, self.n)): ro_frame.columnconfigure(i, weight=1)

        self.ro_vars = []
        for j, coord in enumerate(self.coords[:7]):
            ro_var = tk.StringVar(value="+0.0°")
            self.ro_vars.append(ro_var)
            joint_key = COORD_TO_JOINT.get(coord["name"], None)
            lbl_color = rgb_to_hex(*JOINT_COLORS[joint_key]) if joint_key in JOINT_COLORS else "#8898A8"
            
            item_frame = tk.Frame(ro_frame, bg="#0F1923")
            item_frame.grid(row=0, column=j, padx=4, pady=6, sticky="ew")
            tk.Label(item_frame, text=f"{coord['label']}:", fg=lbl_color, bg="#0F1923", font=("Arial", 8, "bold"), anchor="w").pack(side=tk.LEFT)
            tk.Label(item_frame, textvariable=ro_var, width=6, fg="#FFD43B", bg="#0F1923", anchor="e", font=("Arial", 9, "bold")).pack(side=tk.RIGHT)

        self.root.bind("<r>", lambda e: self._reset_all())
        self.root.bind("<R>", lambda e: self._reset_all())

    def _on_slider(self, idx, val): self.q_tgt[idx] = val
    def _on_speed(self, val): self.speed = val; self.speed_label.configure(text=f"{val:.0f} °/s")
    def _reset_all(self):
        self.q_tgt[:] = 0.0
        for var in self.sl_vars: var.set(0.0)
    def _on_close(self): self.running = False; self.root.destroy()

    def _start_animation_thread(self):
        t = threading.Thread(target=self._animation_loop, daemon=True)
        t.start()

    def _animation_loop(self):
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
                except Exception: pass

            try:
                self.model.realizePosition(self.state)
                # NO MORE MANUAL LINE DRAWING! OpenSim handles the embedded colored axes automatically.
                self.model.getVisualizer().show(self.state)
            except Exception: pass

            self.root.after(0, self._update_ui)
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, self.DT - elapsed))

    def _update_ui(self):
        for j in range(self.n):
            self.val_vars[j].set(f"{self.q[j]:+.0f}°")
            if j < len(self.ro_vars): self.ro_vars[j].set(f"{self.q[j]:+.1f}°")

    def run(self): self.root.mainloop()

if __name__ == "__main__":
    app = DAS3SliderApp()
    app.run()
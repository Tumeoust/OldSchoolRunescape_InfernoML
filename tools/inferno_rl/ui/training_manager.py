"""
Inferno RL Training Manager UI.

A Tkinter-based GUI for managing Inferno RL training runs and visualizer playback.

Usage:
    python -m tools.inferno_rl.ui.training_manager
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List, Optional


class ToolTip:
    """Simple tooltip implementation for Tkinter widgets."""
    
    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
    
    def show(self, event=None):
        if self.tooltip_window:
            return
        
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("TkDefaultFont", 9)
        )
        label.pack()
    
    def hide(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class TrainingManagerApp:
    """Main Tkinter application for managing Inferno RL training."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Inferno RL Training Manager")
        self.root.geometry("650x700")
        self.root.minsize(600, 650)
        
        # Get the workspace root (where we run from)
        self.workspace_root = Path(__file__).parent.parent.parent.parent
        
        # Create main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create tabs
        self.training_tab = ttk.Frame(self.notebook, padding="10")
        self.visualizer_tab = ttk.Frame(self.notebook, padding="10")
        
        self.notebook.add(self.training_tab, text="Training")
        self.notebook.add(self.visualizer_tab, text="Visualizer")
        
        # Build UI
        self._create_training_tab()
        self._create_visualizer_tab()
        self._create_status_bar()
        
        # Populate models after status bar exists
        self._refresh_models()
        
        # Track running processes
        self.training_process: Optional[subprocess.Popen] = None
        self.visualizer_process: Optional[subprocess.Popen] = None
    
    def _create_training_tab(self):
        """Create the training configuration tab."""
        tab = self.training_tab
        row = 0
        
        # === Wave Range Section ===
        wave_frame = ttk.LabelFrame(tab, text="Wave Range", padding="10")
        wave_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        row += 1
        
        ttk.Label(wave_frame, text="Start Wave:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.start_wave_var = tk.StringVar(value="31")
        self.start_wave_entry = ttk.Entry(wave_frame, textvariable=self.start_wave_var, width=10)
        self.start_wave_entry.grid(row=0, column=1, sticky="w", padx=(0, 20))
        ToolTip(self.start_wave_entry, "First wave to train on (1-66)")
        
        ttk.Label(wave_frame, text="Max Wave:").grid(row=0, column=2, sticky="w", padx=(0, 5))
        self.max_wave_var = tk.StringVar(value="66")
        self.max_wave_entry = ttk.Entry(wave_frame, textvariable=self.max_wave_var, width=10)
        self.max_wave_entry.grid(row=0, column=3, sticky="w")
        ToolTip(self.max_wave_entry, "Last wave to train on (episode ends when cleared)")
        
        ttk.Label(wave_frame, text="Mixed Waves:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.mixed_waves_var = tk.StringVar(value="")
        self.mixed_waves_entry = ttk.Entry(wave_frame, textvariable=self.mixed_waves_var, width=35)
        self.mixed_waves_entry.grid(row=1, column=1, columnspan=3, sticky="w", pady=(5, 0))
        ToolTip(self.mixed_waves_entry, "Optional: Mixed start waves (e.g. 50:0.6,35:0.3,1:0.1). Overrides Start Wave.")
        
        # === Training Settings Section ===
        settings_frame = ttk.LabelFrame(tab, text="Training Settings", padding="10")
        settings_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        row += 1
        
        settings_row = 0
        
        # Timesteps
        ttk.Label(settings_frame, text="Timesteps:").grid(row=settings_row, column=0, sticky="w", pady=2)
        self.timesteps_var = tk.StringVar(value="1000000")
        entry = ttk.Entry(settings_frame, textvariable=self.timesteps_var, width=15)
        entry.grid(row=settings_row, column=1, sticky="w", pady=2)
        ToolTip(entry, "Total environment steps to train for")
        settings_row += 1
        
        # Parallel Envs
        ttk.Label(settings_frame, text="Parallel Envs:").grid(row=settings_row, column=0, sticky="w", pady=2)
        self.n_envs_var = tk.StringVar(value="4")
        entry = ttk.Entry(settings_frame, textvariable=self.n_envs_var, width=15)
        entry.grid(row=settings_row, column=1, sticky="w", pady=2)
        ToolTip(entry, "Number of parallel environments (more = faster training, more RAM)")
        settings_row += 1
        
        # Learning Rate
        ttk.Label(settings_frame, text="Learning Rate:").grid(row=settings_row, column=0, sticky="w", pady=2)
        self.lr_var = tk.StringVar(value="0.0001")
        entry = ttk.Entry(settings_frame, textvariable=self.lr_var, width=15)
        entry.grid(row=settings_row, column=1, sticky="w", pady=2)
        ToolTip(entry, "Learning rate for optimizer (smaller = more stable, slower)")
        settings_row += 1
        
        # Entropy Coefficient
        ttk.Label(settings_frame, text="Entropy Coef:").grid(row=settings_row, column=0, sticky="w", pady=2)
        self.ent_coef_var = tk.StringVar(value="0.01")
        entry = ttk.Entry(settings_frame, textvariable=self.ent_coef_var, width=15)
        entry.grid(row=settings_row, column=1, sticky="w", pady=2)
        ToolTip(entry, "Entropy coefficient - higher = more exploration, lower = more exploitation")
        settings_row += 1
        
        # Target KL
        ttk.Label(settings_frame, text="Target KL:").grid(row=settings_row, column=0, sticky="w", pady=2)
        self.target_kl_var = tk.StringVar(value="0.015")
        entry = ttk.Entry(settings_frame, textvariable=self.target_kl_var, width=15)
        entry.grid(row=settings_row, column=1, sticky="w", pady=2)
        ToolTip(entry, "KL divergence threshold for early stopping (prevents policy collapse)")
        settings_row += 1
        
        # Clip Reward
        ttk.Label(settings_frame, text="Clip Reward:").grid(row=settings_row, column=0, sticky="w", pady=2)
        self.clip_reward_var = tk.StringVar(value="10.0")
        entry = ttk.Entry(settings_frame, textvariable=self.clip_reward_var, width=15)
        entry.grid(row=settings_row, column=1, sticky="w", pady=2)
        ToolTip(entry, "Reward clipping threshold after normalization")
        settings_row += 1
        
        # Checkpoint Freq
        ttk.Label(settings_frame, text="Checkpoint Freq:").grid(row=settings_row, column=0, sticky="w", pady=2)
        self.checkpoint_freq_var = tk.StringVar(value="1000000")
        entry = ttk.Entry(settings_frame, textvariable=self.checkpoint_freq_var, width=15)
        entry.grid(row=settings_row, column=1, sticky="w", pady=2)
        ToolTip(entry, "Save checkpoint every N steps")
        settings_row += 1
        
        # Seed
        ttk.Label(settings_frame, text="Seed:").grid(row=settings_row, column=0, sticky="w", pady=2)
        self.seed_var = tk.StringVar(value="42")
        entry = ttk.Entry(settings_frame, textvariable=self.seed_var, width=15)
        entry.grid(row=settings_row, column=1, sticky="w", pady=2)
        ToolTip(entry, "Random seed for reproducibility")
        settings_row += 1
        
        # Reward Normalization checkbox
        self.reward_norm_var = tk.BooleanVar(value=True)
        cb = ttk.Checkbutton(settings_frame, text="Reward Normalization", variable=self.reward_norm_var)
        cb.grid(row=settings_row, column=0, columnspan=2, sticky="w", pady=5)
        ToolTip(cb, "Normalize rewards using running statistics (recommended)")
        settings_row += 1

        # Reward Terms logging checkbox
        self.log_reward_terms_var = tk.BooleanVar(value=False)
        cb = ttk.Checkbutton(settings_frame, text="Log Reward Terms", variable=self.log_reward_terms_var)
        cb.grid(row=settings_row, column=0, columnspan=2, sticky="w", pady=5)
        ToolTip(cb, "Log per-episode raw reward term contributions to TensorBoard (raw_reward_terms/*)")
        
        # === Directories Section ===
        dir_frame = ttk.LabelFrame(tab, text="Directories", padding="10")
        dir_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        row += 1
        
        # Save Dir
        ttk.Label(dir_frame, text="Save Dir:").grid(row=0, column=0, sticky="w", pady=2)
        self.save_dir_var = tk.StringVar(value="tools/models/inferno")
        entry = ttk.Entry(dir_frame, textvariable=self.save_dir_var, width=35)
        entry.grid(row=0, column=1, sticky="w", pady=2, padx=(0, 5))
        ttk.Button(dir_frame, text="Browse", command=self._browse_save_dir, width=8).grid(row=0, column=2)
        
        # Log Dir
        ttk.Label(dir_frame, text="Log Dir:").grid(row=1, column=0, sticky="w", pady=2)
        self.log_dir_var = tk.StringVar(value="tools/logs/inferno")
        entry = ttk.Entry(dir_frame, textvariable=self.log_dir_var, width=35)
        entry.grid(row=1, column=1, sticky="w", pady=2, padx=(0, 5))
        ttk.Button(dir_frame, text="Browse", command=self._browse_log_dir, width=8).grid(row=1, column=2)
        
        # === Continue Training Section ===
        continue_frame = ttk.LabelFrame(tab, text="Continue Training (optional)", padding="10")
        continue_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        row += 1
        
        ttk.Label(continue_frame, text="Load Model:").grid(row=0, column=0, sticky="w", pady=2)
        self.load_model_var = tk.StringVar(value="")
        entry = ttk.Entry(continue_frame, textvariable=self.load_model_var, width=35)
        entry.grid(row=0, column=1, sticky="w", pady=2, padx=(0, 5))
        ToolTip(entry, "Path to existing model .zip to continue training from")
        ttk.Button(continue_frame, text="Browse", command=self._browse_load_model, width=8).grid(row=0, column=2)
        
        # === Action Buttons ===
        button_frame = ttk.Frame(tab)
        button_frame.grid(row=row, column=0, columnspan=2, pady=10)
        
        self.start_btn = ttk.Button(
            button_frame, 
            text="Start Training", 
            command=self._start_training,
            width=20
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(
            button_frame,
            text="Stop Training",
            command=self._stop_training,
            width=15,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Open TensorBoard",
            command=self._open_tensorboard,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
    
    def _create_visualizer_tab(self):
        """Create the visualizer playback tab."""
        tab = self.visualizer_tab
        row = 0
        
        # === Model Selection Section ===
        model_frame = ttk.LabelFrame(tab, text="Model Selection", padding="10")
        model_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        row += 1
        
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky="w", pady=2)
        
        # Model dropdown
        self.viz_model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            model_frame, 
            textvariable=self.viz_model_var, 
            width=45,
            state="readonly"
        )
        self.model_combo.grid(row=0, column=1, sticky="w", pady=2, padx=(0, 5))
        
        ttk.Button(model_frame, text="Refresh", command=self._refresh_models, width=8).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(model_frame, text="Browse", command=self._browse_viz_model, width=8).grid(row=0, column=3)
        
        # === Playback Settings Section ===
        playback_frame = ttk.LabelFrame(tab, text="Playback Settings", padding="10")
        playback_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        row += 1
        
        # Wave range
        ttk.Label(playback_frame, text="Start Wave:").grid(row=0, column=0, sticky="w", pady=2)
        self.viz_start_wave_var = tk.StringVar(value="31")
        entry = ttk.Entry(playback_frame, textvariable=self.viz_start_wave_var, width=10)
        entry.grid(row=0, column=1, sticky="w", pady=2, padx=(0, 20))
        
        ttk.Label(playback_frame, text="Max Wave:").grid(row=0, column=2, sticky="w", pady=2)
        self.viz_max_wave_var = tk.StringVar(value="66")
        entry = ttk.Entry(playback_frame, textvariable=self.viz_max_wave_var, width=10)
        entry.grid(row=0, column=3, sticky="w", pady=2)
        
        # FPS and Tile Size
        ttk.Label(playback_frame, text="FPS:").grid(row=1, column=0, sticky="w", pady=2)
        self.viz_fps_var = tk.StringVar(value="4")
        entry = ttk.Entry(playback_frame, textvariable=self.viz_fps_var, width=10)
        entry.grid(row=1, column=1, sticky="w", pady=2, padx=(0, 20))
        ToolTip(entry, "Playback speed (ticks per second)")
        
        ttk.Label(playback_frame, text="Tile Size:").grid(row=1, column=2, sticky="w", pady=2)
        self.viz_tile_size_var = tk.StringVar(value="20")
        entry = ttk.Entry(playback_frame, textvariable=self.viz_tile_size_var, width=10)
        entry.grid(row=1, column=3, sticky="w", pady=2)
        ToolTip(entry, "Pixel size per tile in visualizer")
        
        # Seed and Episodes
        ttk.Label(playback_frame, text="Seed:").grid(row=2, column=0, sticky="w", pady=2)
        self.viz_seed_var = tk.StringVar(value="")
        entry = ttk.Entry(playback_frame, textvariable=self.viz_seed_var, width=10)
        entry.grid(row=2, column=1, sticky="w", pady=2, padx=(0, 20))
        ToolTip(entry, "Random seed for spawn positions (empty = random each run)")
        
        ttk.Label(playback_frame, text="Episodes:").grid(row=2, column=2, sticky="w", pady=2)
        self.viz_episodes_var = tk.StringVar(value="1")
        entry = ttk.Entry(playback_frame, textvariable=self.viz_episodes_var, width=10)
        entry.grid(row=2, column=3, sticky="w", pady=2)
        
        # Random actions checkbox
        self.viz_random_var = tk.BooleanVar(value=False)
        cb = ttk.Checkbutton(playback_frame, text="Random Actions (ignore model)", variable=self.viz_random_var)
        cb.grid(row=3, column=0, columnspan=4, sticky="w", pady=5)
        ToolTip(cb, "Use random valid actions instead of model predictions")
        
        # === Action Button ===
        button_frame = ttk.Frame(tab)
        button_frame.grid(row=row, column=0, columnspan=2, pady=10)
        
        ttk.Button(
            button_frame,
            text="Launch Visualizer",
            command=self._launch_visualizer,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
    
    def _create_status_bar(self):
        """Create the status bar at the bottom."""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2)
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def _set_status(self, message: str):
        """Update the status bar message."""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    # === Browse Dialogs ===
    
    def _browse_save_dir(self):
        path = filedialog.askdirectory(
            initialdir=str(self.workspace_root / "tools" / "models"),
            title="Select Save Directory"
        )
        if path:
            # Make relative to workspace if possible
            try:
                rel_path = Path(path).relative_to(self.workspace_root)
                self.save_dir_var.set(str(rel_path))
            except ValueError:
                self.save_dir_var.set(path)
    
    def _browse_log_dir(self):
        path = filedialog.askdirectory(
            initialdir=str(self.workspace_root / "tools" / "logs"),
            title="Select Log Directory"
        )
        if path:
            try:
                rel_path = Path(path).relative_to(self.workspace_root)
                self.log_dir_var.set(str(rel_path))
            except ValueError:
                self.log_dir_var.set(path)
    
    def _browse_load_model(self):
        path = filedialog.askopenfilename(
            initialdir=str(self.workspace_root / "tools" / "models" / "inferno"),
            title="Select Model to Continue Training",
            filetypes=[("Model files", "*.zip"), ("All files", "*.*")]
        )
        if path:
            try:
                rel_path = Path(path).relative_to(self.workspace_root)
                self.load_model_var.set(str(rel_path))
            except ValueError:
                self.load_model_var.set(path)
    
    def _browse_viz_model(self):
        path = filedialog.askopenfilename(
            initialdir=str(self.workspace_root / "tools" / "models" / "inferno"),
            title="Select Model for Visualization",
            filetypes=[("Model files", "*.zip"), ("All files", "*.*")]
        )
        if path:
            try:
                rel_path = Path(path).relative_to(self.workspace_root)
                self.viz_model_var.set(str(rel_path))
            except ValueError:
                self.viz_model_var.set(path)
    
    # === Model Discovery ===
    
    def _discover_models(self) -> List[str]:
        """Find all saved model .zip files."""
        models = []
        models_dir = self.workspace_root / "tools" / "models" / "inferno"
        
        if models_dir.exists():
            for zip_file in models_dir.rglob("*.zip"):
                try:
                    rel_path = zip_file.relative_to(self.workspace_root)
                    models.append(str(rel_path))
                except ValueError:
                    models.append(str(zip_file))
        
        return sorted(models, reverse=True)  # Most recent first (by name)
    
    def _refresh_models(self):
        """Refresh the model dropdown."""
        models = self._discover_models()
        self.model_combo["values"] = models
        if models:
            self.model_combo.set(models[0])
        self._set_status(f"Found {len(models)} model(s)")
    
    # === Validation ===
    
    def _validate_training_params(self) -> Optional[str]:
        """Validate training parameters. Returns error message or None."""
        try:
            start_wave = int(self.start_wave_var.get())
            max_wave = int(self.max_wave_var.get())
            if start_wave < 1 or start_wave > 66:
                return "Start wave must be between 1 and 66"
            if max_wave < 1 or max_wave > 66:
                return "Max wave must be between 1 and 66"
            # If mixed waves is NOT used, then start_wave must be <= max_wave
            if not self.mixed_waves_var.get().strip() and start_wave > max_wave:
                return "Start wave must be <= max wave"
        except ValueError:
            return "Wave values must be integers"
            
        mixed_waves = self.mixed_waves_var.get().strip()
        if mixed_waves:
            try:
                # Basic format validation
                for part in mixed_waves.split(","):
                    wave, weight = part.split(":")
                    int(wave)
                    float(weight)
            except ValueError:
                return "Invalid Mixed Waves format. Expected: wave:prob,wave:prob (e.g. 50:0.6,35:0.4)"
        
        try:
            timesteps = int(self.timesteps_var.get())
            if timesteps <= 0:
                return "Timesteps must be positive"
        except ValueError:
            return "Timesteps must be an integer"
        
        try:
            n_envs = int(self.n_envs_var.get())
            if n_envs <= 0:
                return "Parallel envs must be positive"
        except ValueError:
            return "Parallel envs must be an integer"
        
        try:
            lr = float(self.lr_var.get())
            if lr <= 0:
                return "Learning rate must be positive"
        except ValueError:
            return "Learning rate must be a number"
        
        try:
            ent_coef = float(self.ent_coef_var.get())
            if ent_coef < 0:
                return "Entropy coefficient must be non-negative"
        except ValueError:
            return "Entropy coefficient must be a number"
        
        try:
            target_kl = float(self.target_kl_var.get())
            if target_kl <= 0:
                return "Target KL must be positive"
        except ValueError:
            return "Target KL must be a number"
        
        try:
            clip_reward = float(self.clip_reward_var.get())
            if clip_reward <= 0:
                return "Clip reward must be positive"
        except ValueError:
            return "Clip reward must be a number"
        
        load_model = self.load_model_var.get().strip()
        if load_model:
            model_path = self.workspace_root / load_model
            if not model_path.exists():
                return f"Model file not found: {load_model}"
        
        return None
    
    def _validate_visualizer_params(self) -> Optional[str]:
        """Validate visualizer parameters. Returns error message or None."""
        model_path = self.viz_model_var.get().strip()
        if not model_path and not self.viz_random_var.get():
            return "Please select a model or enable random actions"
        
        if model_path:
            full_path = self.workspace_root / model_path
            if not full_path.exists():
                return f"Model file not found: {model_path}"
        
        try:
            start_wave = int(self.viz_start_wave_var.get())
            max_wave = int(self.viz_max_wave_var.get())
            if start_wave > max_wave:
                return "Start wave must be <= max wave"
        except ValueError:
            return "Wave values must be integers"
        
        try:
            fps = int(self.viz_fps_var.get())
            if fps <= 0:
                return "FPS must be positive"
        except ValueError:
            return "FPS must be an integer"
        
        return None
    
    # === Actions ===
    
    def _start_training(self):
        """Start a training run."""
        error = self._validate_training_params()
        if error:
            messagebox.showerror("Validation Error", error)
            return
        
        # Build command
        cmd = [
            sys.executable, "-m", "tools.inferno_rl.training.train",
            "--timesteps", self.timesteps_var.get(),
            "--start-wave", self.start_wave_var.get(),
            "--max-wave", self.max_wave_var.get(),
            "--n-envs", self.n_envs_var.get(),
            "--lr", self.lr_var.get(),
            "--ent-coef", self.ent_coef_var.get(),
            "--target-kl", self.target_kl_var.get(),
            "--clip-reward", self.clip_reward_var.get(),
            "--checkpoint-freq", self.checkpoint_freq_var.get(),
            "--seed", self.seed_var.get(),
            "--save-dir", self.save_dir_var.get(),
            "--log-dir", self.log_dir_var.get(),
        ]
        
        mixed_waves = self.mixed_waves_var.get().strip()
        if mixed_waves:
            cmd.extend(["--mixed-waves", mixed_waves])
        
        load_model = self.load_model_var.get().strip()
        if load_model:
            cmd.extend(["--load", load_model])
        
        if not self.reward_norm_var.get():
            cmd.append("--no-reward-norm")
        if getattr(self, "log_reward_terms_var", None) is not None and self.log_reward_terms_var.get():
            cmd.append("--log-reward-terms")
        
        # Launch in subprocess
        try:
            self.training_process = subprocess.Popen(
                cmd,
                cwd=str(self.workspace_root),
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
            )
            self._set_status(f"Training started (PID: {self.training_process.pid})")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            # Monitor process in background
            threading.Thread(target=self._monitor_training, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {e}")
    
    def _monitor_training(self):
        """Monitor the training process in background."""
        if self.training_process:
            self.training_process.wait()
            self.root.after(0, self._training_finished)
    
    def _training_finished(self):
        """Called when training process finishes."""
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if self.training_process:
            code = self.training_process.returncode
            if code == 0:
                self._set_status("Training completed successfully")
                self._refresh_models()
            else:
                self._set_status(f"Training exited with code {code}")
            self.training_process = None
    
    def _stop_training(self):
        """Stop the running training process."""
        if self.training_process:
            self.training_process.terminate()
            self._set_status("Training stopped")
    
    def _open_tensorboard(self):
        """Open TensorBoard for the log directory."""
        log_dir = self.log_dir_var.get()
        
        try:
            if sys.platform == "win32":
                subprocess.Popen(
                    ["tensorboard", "--logdir", log_dir],
                    cwd=str(self.workspace_root),
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                subprocess.Popen(
                    ["tensorboard", "--logdir", log_dir],
                    cwd=str(self.workspace_root)
                )
            self._set_status("TensorBoard started - open http://localhost:6006")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start TensorBoard: {e}")
    
    def _launch_visualizer(self):
        """Launch the visualizer."""
        error = self._validate_visualizer_params()
        if error:
            messagebox.showerror("Validation Error", error)
            return
        
        # Build command
        cmd = [
            sys.executable, "-m", "tools.inferno_rl.visualizer.run_visual",
            "--start-wave", self.viz_start_wave_var.get(),
            "--max-wave", self.viz_max_wave_var.get(),
            "--fps", self.viz_fps_var.get(),
            "--tile-size", self.viz_tile_size_var.get(),
            "--episodes", self.viz_episodes_var.get(),
        ]
        
        model_path = self.viz_model_var.get().strip()
        if model_path and not self.viz_random_var.get():
            cmd.extend(["--model", model_path])
        
        if self.viz_random_var.get():
            cmd.append("--random")
        
        seed = self.viz_seed_var.get().strip()
        if seed:
            cmd.extend(["--seed", seed])
        
        # Launch in subprocess
        try:
            self.visualizer_process = subprocess.Popen(
                cmd,
                cwd=str(self.workspace_root)
            )
            self._set_status(f"Visualizer launched (PID: {self.visualizer_process.pid})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch visualizer: {e}")
    
    def run(self):
        """Run the application main loop."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = TrainingManagerApp()
    app.run()


if __name__ == "__main__":
    main()

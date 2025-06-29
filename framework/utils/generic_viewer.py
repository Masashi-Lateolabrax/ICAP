import dataclasses
import enum
import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Tuple, Optional
import time
import threading
import logging

from framework.interfaces.backend import SimulatorBackend

from PIL import Image, ImageTk

# Constants
# UIのデフォルト値であるためハードコーディングでよい．
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600
DEFAULT_CAMERA_X = 3.0
DEFAULT_CAMERA_Y = 3.0
DEFAULT_CAMERA_Z = 2.0
DEFAULT_LOOKAT_X = 0.0
DEFAULT_LOOKAT_Y = 0.0
DEFAULT_LOOKAT_Z = 0.0
TARGET_FPS = 30.0
UI_REFRESH_INTERVAL_MS = 16
THREAD_JOIN_TIMEOUT = 1.0


@dataclasses.dataclass
class Position3d:
    x: float
    y: float
    z: float

    def to_tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z


class SimulationRunningMode(enum.Enum):
    RUNNING = "running"
    PAUSED = "paused"
    RESTART = "restart"
    ONE_STEP = "one_step"


@dataclasses.dataclass
class _SimulationState:
    rgb_buffer: np.ndarray
    buffer_update_timestamp: Optional[float] = time.time()

    running_mode: SimulationRunningMode = SimulationRunningMode.RUNNING
    mode_change_event: threading.Event = dataclasses.field(default_factory=threading.Event)

    camera_position: Position3d = dataclasses.field(
        default_factory=lambda: Position3d(DEFAULT_CAMERA_X, DEFAULT_CAMERA_Y, DEFAULT_CAMERA_Z))
    lookat_position: Position3d = dataclasses.field(
        default_factory=lambda: Position3d(DEFAULT_LOOKAT_X, DEFAULT_LOOKAT_Y, DEFAULT_LOOKAT_Z))
    fps: float = 0.0

    error_message: Optional[str] = None
    error_timestamp: Optional[float] = None

    def update_running_mode_by_external(self, order: SimulationRunningMode):
        if self.running_mode != order:
            self.running_mode = order  # 完全な同期処理を目指していないため読み取りにおける競合は無視してよい

            for _ in range(10):
                if self.mode_change_event.wait(timeout=0.1):
                    break
            else:
                logging.getLogger(__name__).warning(
                    f"Timeout waiting for mode change to {order}, resetting event"
                )
            self.mode_change_event.clear()

        return self.running_mode

    def set_error(self, message: str):
        self.error_message = message
        self.error_timestamp = time.time()

    def clear_error(self):
        self.error_message = None
        self.error_timestamp = None

    @property
    def has_recent_error(self) -> bool:
        return (self.error_message is not None and
                self.error_timestamp is not None and
                time.time() - self.error_timestamp < 5.0)


class _Simulation:
    def __init__(self, backend: SimulatorBackend, state: _SimulationState):
        self.backend = backend
        self.state = state
        self.logger = logging.getLogger(__name__)

        self.thread: Optional[threading.Thread] = None
        self.should_stop = False
        self._running = False
        self._average_elapsed = 0.0

    def start(self):
        if not self._running:
            self.should_stop = False
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            self._running = True

    def stop(self):
        if self._running:
            self.should_stop = True
            if self.thread:
                self.thread.join(timeout=THREAD_JOIN_TIMEOUT)
                if self.thread.is_alive():
                    self.logger.warning("Simulation thread did not stop gracefully")
            self._running = False

    def _render_frame(self):
        try:
            self.backend.render(
                self.state.rgb_buffer,
                self.state.camera_position.to_tuple(),
                self.state.lookat_position.to_tuple()
            )
            self.state.buffer_update_timestamp = time.time()
        except Exception as e:
            self.logger.error(f"Render error: {e}")
            self.state.set_error(f"Render error: {e}")
            self.state.update_running_mode_by_external(SimulationRunningMode.PAUSED)

    def _run_loop(self):
        target_interval = 1.0 / TARGET_FPS
        running_mode = self.state.running_mode

        try:
            while not self.should_stop:
                start_time = time.perf_counter()

                if running_mode != self.state.running_mode:
                    match self.state.running_mode:
                        case SimulationRunningMode.RESTART:
                            self.backend.reset()
                            running_mode = SimulationRunningMode.PAUSED

                        case SimulationRunningMode.ONE_STEP:
                            self.backend.step()
                            running_mode = SimulationRunningMode.PAUSED

                        case SimulationRunningMode.RUNNING:
                            running_mode = SimulationRunningMode.RUNNING

                        case SimulationRunningMode.PAUSED:
                            running_mode = SimulationRunningMode.PAUSED

                    self.state.running_mode = running_mode
                    self.state.mode_change_event.set()

                elif running_mode == SimulationRunningMode.RUNNING:
                    self.backend.step()

                sleep_time = max(0.0, target_interval - self._average_elapsed)
                if sleep_time > 0:
                    self._render_frame()
                    time.sleep(sleep_time)

                elapsed = time.perf_counter() - start_time
                self._average_elapsed = (self._average_elapsed * 0.9) + (elapsed * 0.1)

                if self._average_elapsed > 0:
                    self.state.fps = 1.0 / self._average_elapsed


        except Exception as e:
            self.logger.critical(f"Critical error in simulation loop: {e}")
            self.state.set_error(f"Critical simulation error: {e}")


class SimulationControlPanel(tk.Frame):
    def __init__(self, parent, state: _SimulationState):
        super().__init__(parent)
        self.state = state

        # Set up UI elements
        ttk.Label(self, text="Simulation Control", font=('Arial', 12, 'bold')).pack(pady=(0, 10))

        self.pause_button = ttk.Button(self, text="Pause")
        self.pause_button.pack(fill=tk.X, pady=2)

        step_button = ttk.Button(self, text="Step")
        step_button.pack(fill=tk.X, pady=2)

        reset_button = ttk.Button(self, text="Reset")
        reset_button.pack(fill=tk.X, pady=2)

        # Set up callbacks
        self.pause_button.config(command=self._handle_pause)
        step_button.config(command=self._handle_step)
        reset_button.config(command=self._handle_reset)

        self._update_pause_button_text()

    def _update_pause_button_text(self):
        if self.state.running_mode == SimulationRunningMode.RUNNING:
            self.pause_button.config(text="Pause")
        else:
            self.pause_button.config(text="Resume")

    def _handle_pause(self):
        if self.state.running_mode == SimulationRunningMode.RUNNING:
            self.state.update_running_mode_by_external(SimulationRunningMode.PAUSED)
        else:
            self.state.update_running_mode_by_external(SimulationRunningMode.RUNNING)
        self._update_pause_button_text()

    def _handle_step(self):
        self.state.update_running_mode_by_external(SimulationRunningMode.ONE_STEP)

    def _handle_reset(self):
        self.state.update_running_mode_by_external(SimulationRunningMode.RESTART)
        self.state.clear_error()

    def update(self):
        self._update_pause_button_text()


class CameraControlPanel(ttk.Frame):
    def __init__(self, parent_frame: ttk.Frame, state: _SimulationState):
        super().__init__(parent_frame)
        self.state = state

        self.cam_x_var = tk.DoubleVar(value=DEFAULT_CAMERA_X)
        self.cam_y_var = tk.DoubleVar(value=DEFAULT_CAMERA_Y)
        self.cam_z_var = tk.DoubleVar(value=DEFAULT_CAMERA_Z)
        self.lookat_x_var = tk.DoubleVar(value=DEFAULT_LOOKAT_X)
        self.lookat_y_var = tk.DoubleVar(value=DEFAULT_LOOKAT_Y)
        self.lookat_z_var = tk.DoubleVar(value=DEFAULT_LOOKAT_Z)

        # Panel title
        ttk.Label(self, text="Camera Control", font=('Arial', 12, 'bold')).pack(pady=(0, 10))

        # Camera position controls
        ttk.Label(self, text="Camera Position:", font=('Arial', 10, 'bold')).pack(pady=(10, 5))

        for label, var, range_min, range_max in [
            ("X:", self.cam_x_var, -10.0, 10.0),
            ("Y:", self.cam_y_var, -10.0, 10.0),
            ("Z:", self.cam_z_var, 0.1, 10.0)
        ]:
            ttk.Label(self, text=label).pack()
            pos_scale = ttk.Scale(
                self, from_=range_min, to=range_max, variable=var, orient=tk.HORIZONTAL,
                command=self._handle_camera_change
            )
            pos_scale.pack(fill=tk.X, pady=2)

        # LookAt controls
        ttk.Label(self, text="Look At:", font=('Arial', 10, 'bold')).pack(pady=(15, 5))

        for label, var, range_min, range_max in [
            ("X:", self.lookat_x_var, -5.0, 5.0),
            ("Y:", self.lookat_y_var, -5.0, 5.0),
            ("Z:", self.lookat_z_var, -2.0, 2.0)
        ]:
            ttk.Label(self, text=label).pack()
            scale = ttk.Scale(
                self, from_=range_min, to=range_max, variable=var, orient=tk.HORIZONTAL,
                command=self._handle_camera_change
            )
            scale.pack(fill=tk.X, pady=2)

    def _handle_camera_change(self, value=None):
        self.state.camera_position = Position3d(
            self.cam_x_var.get(),
            self.cam_y_var.get(),
            self.cam_z_var.get()
        )
        self.state.lookat_position = Position3d(
            self.lookat_x_var.get(),
            self.lookat_y_var.get(),
            self.lookat_z_var.get()
        )


class SimulationInfoPanel(ttk.LabelFrame):
    def __init__(self, parent, state: _SimulationState, backend_name: str = "Unknown"):
        super().__init__(parent, text="Simulation Info")
        self.state = state

        self.fps_label = ttk.Label(self, text="FPS: --")
        self.fps_label.pack(pady=2)

        ttk.Label(self, text=f"Backend: {backend_name}").pack(pady=2)

    def update(self):
        self.fps_label.config(text=f"FPS: {self.state.fps:.1f}")


class ControlPanel(ttk.Frame):
    def __init__(self, parent, state: _SimulationState, backend_name: str = "Unknown"):
        super().__init__(parent)
        self.state = state

        self.simulation_panel = SimulationControlPanel(self, state)
        self.simulation_panel.pack(fill=tk.X, pady=(0, 10))

        self.camera_panel = CameraControlPanel(self, state)
        self.camera_panel.pack(fill=tk.X, pady=(0, 10))

        self.info_panel = SimulationInfoPanel(self, state, backend_name)
        self.info_panel.pack(fill=tk.X, pady=(0, 10))

    def update(self):
        self.info_panel.update()
        self.simulation_panel.update()


class _SimulationFrame(tk.Frame):
    def __init__(self, parent, width, height, state: _SimulationState):
        super().__init__(parent)

        self.width = width
        self.height = height

        self.state = state
        self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._cached_img = Image.fromarray(self.state.rgb_buffer)
        self._cached_photo = None
        self.buffer_update_timestamp = state.buffer_update_timestamp

        self.update_idletasks()
        self.pack(fill=tk.BOTH, expand=True)

    def display_image(self):
        try:
            current_timestamp = self.state.buffer_update_timestamp
            if current_timestamp and self.buffer_update_timestamp != current_timestamp:
                self.buffer_update_timestamp = current_timestamp

                if self._cached_img.size == (self.state.rgb_buffer.shape[1], self.state.rgb_buffer.shape[0]):
                    self._cached_img = self._cached_img._new(Image.fromarray(self.state.rgb_buffer).im)
                else:
                    self._cached_img = Image.fromarray(self.state.rgb_buffer)

                self._cached_photo = ImageTk.PhotoImage(self._cached_img)

                self.canvas.delete("all")
                self.canvas.create_image(self.width // 2, self.height // 2, image=self._cached_photo)
                self.canvas.update_idletasks()
        except Exception as e:
            logging.getLogger(__name__).error(f"Display image error: {e}")
            self.canvas.delete("all")
            self.canvas.create_text(
                self.width // 2, self.height // 2,
                text=f"Display error:\n{str(e)}",
                fill="red", font=('Arial', 12)
            )

    def update(self):
        if self.state.has_recent_error:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.width // 2, self.height // 2,
                text=f"Error:\n{self.state.error_message}",
                fill="red", font=('Arial', 12), width=self.width - 20
            )
        else:
            self.display_image()


class _TopWindow(tk.Tk):
    def __init__(self, state: _SimulationState, width, height, backend_name: str = "Unknown"):
        super().__init__()

        self.state = state
        self.logger = logging.getLogger(__name__)

        self._setup_ui(width, height, backend_name)
        self._schedule_ui_update()

    def _setup_ui(self, width, height, backend_name: str):
        self.title("Generic Simulator Viewer")

        self.simulation_frame = _SimulationFrame(self, width, height, self.state)
        self.simulation_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.control_panel = ControlPanel(self, self.state, backend_name)
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)

    def _schedule_ui_update(self):
        try:
            self.simulation_frame.update()
            self.control_panel.update()
            self.after(UI_REFRESH_INTERVAL_MS, self._schedule_ui_update)
        except Exception as e:
            self.logger.error(f"UI update error: {e}")


class GenericTkinterViewer:
    def __init__(self, backend: SimulatorBackend, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        self.backend = backend
        self.logger = logging.getLogger(__name__)

        self.state = _SimulationState(
            np.zeros((height, width, 3), dtype=np.uint8)
        )

        self.simulation = _Simulation(backend, self.state)

        backend_name = getattr(backend, '__class__', type(backend)).__name__
        self._viewer = _TopWindow(self.state, width, height, backend_name)

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def run(self):
        """Run the viewer with proper cleanup"""
        try:
            self.simulation.start()
            self._viewer.mainloop()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Runtime error: {e}")

import dataclasses
import math
import threading
from typing import Optional
import torch
from ..prelude import *


@dataclasses.dataclass
class SensorOutputPair:
    robot_sensor: SensorInterface
    food_sensor: SensorInterface
    nest_sensor: SensorInterface
    output_buf: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    need_calculation: bool = False
    is_calculating: bool = False
    finished: bool = False
    error: Optional[Exception] = None


class Controller:
    def __init__(self, settings: Settings, batch_size: int, nn: torch.nn.Module):
        self.settings = settings
        self.batch_size = batch_size
        self.device = next(nn.parameters()).device if list(nn.parameters()) else torch.device('cpu')

        self.input_ndarray = np.zeros((self.batch_size, 3 * 2), dtype=np.float32)
        self.output_ndarray = np.zeros((self.batch_size, 2), dtype=np.float32)

        if self.device.type == 'cpu':
            self.input_tensor_cpu = torch.from_numpy(self.input_ndarray)
            self.output_tensor_cpu = torch.from_numpy(self.output_ndarray)
        else:
            self.input_tensor_cpu = torch.from_numpy(self.input_ndarray)
            self.output_tensor_cpu = torch.from_numpy(self.output_ndarray)
            self.input_tensor_device = torch.zeros(
                self.input_tensor_cpu.shape, dtype=torch.float32, device=self.device
            )
            self.output_tensor_device = torch.zeros(
                self.output_tensor_cpu.shape, dtype=torch.float32, device=self.device
            )

        self._pairs: list[SensorOutputPair] = []
        self._pairs_lock = threading.RLock()

        self.nn = nn
        self.nn.eval()

    @property
    def input_dim(self):
        return self.input_ndarray.shape[1]

    @property
    def output_dim(self):
        return self.output_ndarray.shape[1]

    def register(
            self, robot_sensor: SensorInterface, food_sensor: SensorInterface, nest_sensor: SensorInterface
    ) -> SensorOutputPair:
        pair = SensorOutputPair(
            robot_sensor=robot_sensor,
            food_sensor=food_sensor,
            nest_sensor=nest_sensor,
        )
        self._pairs.append(pair)
        return pair

    def _remove_finished(self):
        with self._pairs_lock:
            i = 0
            while i < len(self._pairs):
                if self._pairs[i].finished:
                    self._pairs.pop(i)
                    continue
                i += 1

    def _collect_sensor_data(self, batch: list[SensorOutputPair]) -> list[SensorOutputPair]:
        valid_pairs = []
        for i, p in enumerate(batch):
            try:
                robot_data = p.robot_sensor.get()
                food_data = p.food_sensor.get()
                nest_data = p.nest_sensor.get()

                # Validate sensor data
                if not (isinstance(robot_data, np.ndarray) and robot_data.shape == (2,)):
                    raise ValueError(f"Invalid robot sensor data shape: {robot_data.shape}")
                if not (isinstance(food_data, np.ndarray) and food_data.shape == (2,)):
                    raise ValueError(f"Invalid food sensor data shape: {food_data.shape}")
                if not (isinstance(nest_data, np.ndarray) and nest_data.shape == (2,)):
                    raise ValueError(f"Invalid nest sensor data shape: {nest_data.shape}")

                self.input_ndarray[i, 0:2] = robot_data
                self.input_ndarray[i, 2:4] = food_data
                self.input_ndarray[i, 4:6] = nest_data
                valid_pairs.append(p)
            except Exception as e:
                print(f"Sensor data collection error for pair {i}: {e}")
                p.error = e
                p.is_calculating = False
                # Set default values to prevent NaN propagation
                self.input_ndarray[i, :] = 0.0
        return valid_pairs

    def _run_inference(self, batch_size: int):
        if self.device.type == 'cpu':
            with torch.no_grad():
                output = self.nn(self.input_tensor_cpu[:batch_size])
                self.output_tensor_cpu[:batch_size] = output
        else:
            self.input_tensor_device[:batch_size].copy_(self.input_tensor_cpu[:batch_size])

            with torch.no_grad():
                output = self.nn(self.input_tensor_device[:batch_size])
                self.output_tensor_device[:batch_size] = output

            self.output_tensor_cpu[:batch_size].copy_(self.output_tensor_device[:batch_size].cpu())

    def _process_batch(self, batch: list[SensorOutputPair]):
        for p in batch:
            p.is_calculating = True
            p.error = None

        try:
            valid_pairs = self._collect_sensor_data(batch)
            if not valid_pairs:
                return

            self._run_inference(len(valid_pairs))

            valid_index = 0
            for i, p in enumerate(batch):
                if p.error is None:
                    p.output_buf[:] = self.output_ndarray[valid_index, :]
                    p.need_calculation = False
                    valid_index += 1
                p.is_calculating = False

        except Exception as e:
            for p in batch:
                p.error = e
                p.is_calculating = False

    def calculate(self):
        self._remove_finished()

        with self._pairs_lock:
            pairs = [p for p in self._pairs if p.need_calculation and not p.is_calculating]

        if not pairs:
            return

        for b in range(math.ceil(len(pairs) / self.batch_size)):
            batch = pairs[b * self.batch_size:min((b + 1) * self.batch_size, len(pairs))]
            self._process_batch(batch)

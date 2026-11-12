"""
Defines HybridIoTDataset: the abstract base class for all datasets in the
HybridIoT cooperative multi-agent estimation system.

The core data is stored as a 3D PyTorch tensor:
    values: torch.Tensor, shape (n_types, n_timesteps, n_sensors)
    axis 0: data type index  (e.g. 0=temperature, 1=humidity, ...)
    axis 1: timestep index   (uniformly spaced at sensors_aggr_frequency)
    axis 2: sensor index     (one column per sensing station)

This layout makes slicing by type, time, or sensor a single tensor
operation, and integrates directly with PyTorch DataLoader.

PyTorch
__getitem__(index) returns a (input, target) pair for ML training:
  input: tensor (n_types, window_size, n_sensors) = the input window
  target: tensor (n_types, n_sensors) = the next timestep

The model learns: "given these window_size past measurements across all
sensors and types, predict the next timestep."
window_size is set at construction time (default 96 = 24h at 15min).

For structured windowed access with metadata, use get_data_window() instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import torch
from torch.utils.data import Dataset as TorchDataset

from .sensor import Sensor
from .data_window import DataWindow


@dataclass
class HybridIoTDataset(TorchDataset, ABC):
    """
    Abstract base class for HybridIoT datasets.

    sensors : List[Sensor]
        All sensing agents in this dataset, in tensor column order.
    values : torch.Tensor, shape (n_types, n_timesteps, n_sensors)
        Core data tensor. NaN = missing or anomalous value.
    data_types : List[str]
        Names of the data types in axis-0 order
        (e.g. ["temperature", "humidite", "pression"]).
    start_time : datetime
        Timestamp of the first timestep (axis 1, index 0).
    end_time : datetime
        Timestamp of the last timestep.
    sensors_aggr_frequency : int
        Sampling frequency in minutes (e.g. 15 for Toulouse Open Data).
        Automatically computed from the CSV in from_csv().
    window_size : int
        Number of consecutive timesteps per ML training sample.
        Default 96 = 24h at 15-minute frequency.
        __getitem__(i) returns values[:, i:i+window_size, :] as input
        and values[:, i+window_size, :] as target.
    name : str
        Human-readable name for this dataset.
    """

    sensors:                 List[Sensor]
    values:                  torch.Tensor        # (n_types, n_timesteps, n_sensors)
    data_types:              List[str]
    start_time:              datetime
    end_time:                datetime
    sensors_aggr_frequency:  int = 15
    window_size:             int = 96
    name:                    str = "unnamed"

    @property
    def n_types(self) -> int:
        return self.values.shape[0]

    @property
    def n_timesteps(self) -> int:
        return self.values.shape[1]

    @property
    def n_sensors(self) -> int:
        return self.values.shape[2]

    @property
    def timestamps(self) -> List[datetime]:
        """Full list of timestamps, one per timestep."""
        freq = timedelta(minutes=self.sensors_aggr_frequency)
        return [self.start_time + i * freq for i in range(self.n_timesteps)]

    def _type_index(self, data_type: str) -> int:
        """Return the tensor axis-0 index for a data type name."""
        if data_type not in self.data_types:
            raise KeyError(
                f"Unknown data type '{data_type}'. "
                f"Available: {self.data_types}"
            )
        return self.data_types.index(data_type)

    def _sensor_index(self, sensor_id: str) -> int:
        """Return the tensor axis-2 index for a sensor ID."""
        for i, s in enumerate(self.sensors):
            if s.id == sensor_id:
                return i
        raise KeyError(
            f"Unknown sensor '{sensor_id}'. "
            f"Available: {[s.id for s in self.sensors]}"
        )
    
    def get_sensor(self, sensor_id: str) -> Optional[Sensor]:
        """Return the Sensor with the given ID, or None if not found."""
        for sensor in self.sensors:
            if sensor.id == sensor_id:
                return sensor
        return None

    def _time_slice(
        self,
        start: Optional[datetime],
        end:   Optional[datetime],
    ) -> tuple[int, int, List[datetime]]:
        """
        Convert a (start, end) datetime range to tensor axis-1 slice indices.
        Returns (i_start, i_end, timestamps_in_range).
        None means no bound on that side.
        Returns an empty slice (0, 0, []) if the range is outside the dataset.
        """
        all_ts = self.timestamps

        i_start = 0
        i_end   = self.n_timesteps

        if start is not None:
            # Find first timestamp >= start
            found = False
            for i, ts in enumerate(all_ts):
                if ts >= start:
                    i_start = i
                    found = True
                    break
            if not found:
                # start is after all timestamps -> empty window
                return 0, 0, []

        if end is not None:
            # Find last timestamp <= end
            found = False
            for i in range(self.n_timesteps - 1, -1, -1):
                if all_ts[i] <= end:
                    i_end = i + 1
                    found = True
                    break
            if not found:
                # end is before all timestamps -> empty window
                return 0, 0, []

        if i_start >= i_end:
            return 0, 0, []

        return i_start, i_end, all_ts[i_start:i_end]


    # PyTorch interface 

    def __len__(self) -> int:
        """
        Number of (input, target) pairs available for ML training.
        Each pair uses window_size timesteps as input and 1 as target,
        so the total is n_timesteps - window_size.
        Example: 35040 timesteps, window_size=96 -> 34944 training samples.
        """
        return self.n_timesteps - self.window_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a (input, target) pair for ML training.

        index : int
            Start index of the input window (0-based).
            Valid range: [0, n_timesteps - window_size - 1]

        Returns
        input : torch.Tensor, shape (n_types, window_size, n_sensors)
            The input window: window_size consecutive timesteps.
        target : torch.Tensor, shape (n_types, n_sensors)
            The timestep immediately after the input window.
            This is what the model must learn to predict.

        NaN values are replaced by 0.0 in both tensors.
        """
        x = self.values[:, index : index + self.window_size, :]  # (n_types, W, n_sensors)
        y = self.values[:, index + self.window_size, :]          # (n_types, n_sensors)

        x = torch.nan_to_num(x, nan=0.0)
        y = torch.nan_to_num(y, nan=0.0)

        return x, y


    # Data window for HybridIoT estimation

    def get_data_window(
        self,
        data_types:  Optional[object]    = None,
        start_time:  Optional[datetime]  = None,
        end_time:    Optional[datetime]  = None,
        sensor_ids:  Optional[object]    = None,
    ) -> DataWindow:
        """
        Extract a sub-tensor from the dataset as a DataWindow.
        Each parameter is optional, unspecified dimensions are not filtered.
        The result tensor is always 3D: (n_types_w, n_timesteps_w, n_sensors_w).

        data_types : str or list of str, optional
            Filter to one or more data types.
            A single string is accepted as convenience.
            None = all types included.
        start_time : datetime, optional
            Start of the time range (inclusive). None = dataset start.
        end_time : datetime, optional
            End of the time range (inclusive). None = dataset end.
        sensor_ids : str or list of str, optional
            Filter to one or more sensors.
            A single string is accepted as convenience.
            None = all sensors included.

        Returns
        DataWindow
            valeurs shape: (n_types_w, n_timesteps_w, n_sensors_w), always 3D.
        """
        # Normalize data_types: None -> all, str -> [str], list -> list
        if data_types is None:
            selected_types = self.data_types
        elif isinstance(data_types, str):
            selected_types = [data_types]
        else:
            selected_types = list(data_types)

        # Normalize sensor_ids: None -> all, str -> [str], list -> list
        if sensor_ids is None:
            selected_sensor_ids = [s.id for s in self.sensors]
        elif isinstance(sensor_ids, str):
            selected_sensor_ids = [sensor_ids]
        else:
            selected_sensor_ids = list(sensor_ids)

        # Filter by data types (axis 0) 
        t_indices = [self._type_index(dt) for dt in selected_types]
        tensor    = self.values[t_indices, :, :]   # (n_types_w, n_timesteps, n_sensors)

        # Filter by time range (axis 1) 
        i_start, i_end, window_ts = self._time_slice(start_time, end_time)
        tensor = tensor[:, i_start:i_end, :]       # (n_types_w, n_timesteps_w, n_sensors)

        # Filter by sensors (axis 2) 
        s_indices = [self._sensor_index(sid) for sid in selected_sensor_ids]
        tensor    = tensor[:, :, s_indices]        # (n_types_w, n_timesteps_w, n_sensors_w)

        return DataWindow(
            type         = ",".join(selected_types),
            valeurs      = tensor,
            interv_temps = (
                window_ts[0]  if window_ts else self.start_time,
                window_ts[-1] if window_ts else self.end_time,
            ),
            sensor_ids   = selected_sensor_ids,
            timestamps   = window_ts,
        )
    

    # State history and fault rate

    def get_state_history(
        self,
        sensor:   Sensor,
        feature:  Optional[str] = None,
    ) -> List[tuple]:
        """
        Derive StateHistory from the dataset tensor for a given sensor.

        feature : str, optional
            If given, returns history only for that feature.
            If None, returns history for all features.

        Returns
        List of (datetime, str, bool): (timestamp, feature_name, is_valid)
        """
        s_idx    = next(i for i, s in enumerate(self.sensors) if s.id == sensor.id)
        all_ts   = self.timestamps
        features = [feature] if feature else self.data_types

        history = []
        for f_name in features:
            f_idx = self.data_types.index(f_name)
            for t_idx, ts in enumerate(all_ts):
                is_valid = not torch.isnan(self.values[f_idx, t_idx, s_idx]).item()
                history.append((ts, f_name, is_valid))

        return history

    def fault_rate_for_sensor(self, sensor: Sensor, feature: Optional[str] = None) -> float:
        """
        Fraction of NaN values for a sensor (optionally for a specific feature).
        0.0 = all valid, 1.0 = all NaN.
        """
        s_idx = next(i for i, s in enumerate(self.sensors) if s.id == sensor.id)
        if feature is not None:
            f_idx  = self.data_types.index(feature)
            col    = self.values[f_idx, :, s_idx]
            return torch.isnan(col).float().mean().item()
        else:
            cube = self.values[:, :, s_idx]
            return torch.isnan(cube).float().mean().item()

    # Quality report 

    def quality_report(self) -> str:
        """Human-readable quality report, NaN counted per (feature, sensor)."""
        lines = [
            f"\nQuality report {self.name}",
            f"  Sensors    : {self.n_sensors}",
            f"  Data types : {self.data_types}",
            f"  Timesteps  : {self.n_timesteps} "
            f"(every {self.sensors_aggr_frequency} min)",
            f"  Time range : {self.start_time} -> {self.end_time}",
            f"  Tensor     : {tuple(self.values.shape)}  "
            f"(n_types, n_timesteps, n_sensors)",
            f"  NaN rate   : {torch.isnan(self.values).float().mean().item():.2%}",
            "  " + "-" * 60,
        ]

        nan_mask = torch.isnan(self.values)   # (n_types, n_timesteps, n_sensors)

        for s_idx, sensor in enumerate(self.sensors):
            overall_nan = self.fault_rate_for_sensor(sensor)
            lines.append(f"\n  [{sensor.id}]  (overall NaN rate: {overall_nan:.1%})")
            for t_idx, dtype in enumerate(self.data_types):
                n_nan = int(nan_mask[t_idx, :, s_idx].sum().item())
                pct   = n_nan / self.n_timesteps * 100
                lines.append(f"    {dtype:<25} NaN: {n_nan:>5} ({pct:>5.1f}%)")

        return "\n".join(lines)

    # Abstract interface 

    @classmethod
    @abstractmethod
    def from_csv(cls, path: str, **kwargs) -> "HybridIoTDataset":
        """Construct the dataset from a CSV file."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable description of this dataset."""
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"types={self.data_types}, "
            f"shape={tuple(self.values.shape)}, "
            f"freq={self.sensors_aggr_frequency}min)"
        )
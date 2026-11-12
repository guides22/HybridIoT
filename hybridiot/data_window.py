"""
Defines DataWindow: a slice of a HybridIoTDataset returned by get_data_window().

A DataWindow is a sub-tensor of the full dataset, filtered by:
  - data type   (e.g. "temperature")
  - time range  (start_time : end_time)
  - sensors     (subset of stations)

It carries enough metadata to be self-describing and to be fed directly
into the HybridIoT estimation system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

import torch


@dataclass
class DataWindow:
    """
    A windowed slice of a HybridIoTDataset.

    type : str
        The data type contained in this window (e.g. "temperature").
        Empty string if the window contains all types (full tensor slice).
    valeurs : torch.Tensor
        The data tensor for this window.
        Shape depends on the filters applied:
          - Single type  : (n_timesteps, n_sensors)
          - All types    : (n_types, n_timesteps, n_sensors)
    interv_temps : tuple[datetime, datetime]
        (start_time, end_time) of this window.
    sensor_ids : List[str]
        IDs of the sensors included in this window, in tensor column order.
    timestamps : List[datetime]
        Timestamps included in this window, in tensor row order.
    """

    type:         str
    valeurs:      torch.Tensor
    interv_temps: tuple
    sensor_ids:   List[str] = field(default_factory=list)
    timestamps:   List[datetime] = field(default_factory=list)

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps in this window."""
        return len(self.timestamps)

    @property
    def n_sensors(self) -> int:
        """Number of sensors in this window."""
        return len(self.sensor_ids)

    @property
    def is_empty(self) -> bool:
        """True if this window has no timesteps at all (time range outside dataset)."""
        return len(self.timestamps) == 0

    @property
    def nan_rate(self) -> float:
        """Fraction of NaN values in the tensor. 0.0 = all valid, 1.0 = all NaN."""
        if self.is_empty:
            return 1.0
        import torch
        return torch.isnan(self.valeurs).float().mean().item()

    @property
    def is_all_nan(self) -> bool:
        """True if every value in the tensor is NaN (data exists but all missing)."""
        return self.nan_rate == 1.0

    def coverage_per_sensor(self, data_type: str = None) -> dict:
        """
        Returns the fraction of valid (non-NaN) values per sensor.

        data_type : str, optional
            If given, computes coverage only for that data type.
            Must be one of the types present in this window (see self.type).
            If None, averages coverage across all types in the window.

        Returns
        dict {sensor_id: float}. 1.0 = fully valid, 0.0 = all NaN
        """
        import torch
        if self.is_empty:
            return {sid: 0.0 for sid in self.sensor_ids}

        # valeurs shape: (n_types_w, n_timesteps_w, n_sensors_w)
        if data_type is not None:
            types_in_window = self.type.split(",")
            if data_type not in types_in_window:
                raise KeyError(
                    f"'{data_type}' not in this window. "
                    f"Available: {types_in_window}"
                )
            t_idx  = types_in_window.index(data_type)
            tensor = self.valeurs[t_idx, :, :]          # (n_timesteps_w, n_sensors_w)
            valid  = (~torch.isnan(tensor)).float()
            per_sensor = valid.mean(dim=0)              # (n_sensors_w,)
        else:
            valid      = (~torch.isnan(self.valeurs)).float()
            per_sensor = valid.mean(dim=(0, 1))         # (n_sensors_w,)

        return {
            sid: per_sensor[i].item()
            for i, sid in enumerate(self.sensor_ids)
        }

    @property
    def duration_minutes(self) -> float:
        """Duration of this window in minutes. Returns 0.0 if empty."""
        if len(self.timestamps) < 2:
            return 0.0
        delta = self.timestamps[-1] - self.timestamps[0]
        return delta.total_seconds() / 60.0

    def __repr__(self) -> str:
        if self.is_empty:
            return f"DataWindow(EMPTY: time range outside dataset)"
        return (
            f"DataWindow("
            f"type={self.type!r}, "
            f"shape={tuple(self.valeurs.shape)}, "
            f"sensors={self.n_sensors}, "
            f"timesteps={self.n_timesteps}, "
            f"duration={self.duration_minutes:.0f}min)"
        )
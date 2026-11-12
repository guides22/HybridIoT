"""
Defines Sensor: a physical or virtual sensing device.

StateHistory is NOT stored in Sensor, it is derived on demand from the
dataset tensor via HybridIoTDataset.get_state_history(sensor).
This avoids the "one bad feature poisons the whole timestamp" problem:
fault detection is per (feature, timestamp), not per row.
For the agent, the situation is different: it is a single (feature, sensor) pair, so StateHistory is stored in the Agent.
"""

from __future__ import annotations
from dataclasses import dataclass

import sys, os
from .position import Position


@dataclass
class Sensor:
    """
    A sensing device (physical or virtual).

    id : str
        Unique identifier (e.g. station name from Open Data).
    position : Position
        Geospatial position (longitude, latitude, altitude, X, Y).
    is_virtual : bool
        True if values are estimated, not physically measured.

    Note: fault detection (StateHistory) is computed from the dataset
    tensor via HybridIoTDataset.get_state_history(sensor, feature).
    """

    id:         str
    position:   Position
    is_virtual: bool = False

    def __repr__(self) -> str:
        return (
            f"Sensor(id={self.id!r}, virtual={self.is_virtual}, "
            f"pos=({self.position.latitude:.4f}, {self.position.longitude:.4f}, {self.position.altitude}m))"   
        )

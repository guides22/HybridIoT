"""
Data abstraction layer for the HybridIoT cooperative multi-agent system.
Sensor              Sensing agent with position and altitude
DataWindow          Windowed slice of a dataset (result of get_data_window)
HybridIoTDataset    Abstract base: 3D tensor + PyTorch DataLoader support
MeteoDataset        Meteorological dataset, built from Toulouse Open Data CSV
Agent               Active sensing agent with local dataset and fault detection
Environment         Shared environment, mediator of all agent communication or local environment
Estimator           Estimation of missing data, hybridiot methods
Position            longitude and latitude, 2D position (x, y) in meters
"""

from .sensor           import Sensor
from .data_window      import DataWindow
from .base_dataset     import HybridIoTDataset
from .meteo_dataset    import MeteoDataset
from .agent            import Agent
from .environment      import Environment
from .estimator        import Estimator
from .position         import Position 

__all__ = [
    "Sensor",
    "DataWindow",
    "HybridIoTDataset",
    "MeteoDataset",
    "Agent",
    "Environment",
    "Estimator",
    "Position"
]
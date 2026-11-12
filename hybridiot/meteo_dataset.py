"""
Defines MeteoDataset: concrete implementation of HybridIoTDataset for meteorological
sensor networks (Toulouse Open Data format).

Parsing pipeline
1. Read CSV produced by download_toulouse_meteo.py
2. For each station -> build a Sensor (id, position, altitude)
3. Build the 3D tensor (n_types, n_timesteps, n_sensors):
     axis 0 : data type  (temperature, humidite, pression, pluie, ...)
     axis 1 : timestep   (uniformly spaced, frequency computed from CSV)
     axis 2 : sensor     (one column per station)
4. Anomalous values (outside valid physical ranges) are set to NaN in the
   tensor. Fault detection is available via get_state_history() in
   HybridIoTDataset, no StateHistory stored in Sensor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import List

import pandas as pd
import torch

import sys
from .position import Position
from .sensor import Sensor
from .base_dataset import HybridIoTDataset


# Values outside these ranges are considered anomalous -> set to NaN in tensor.
# Fault detection is derived from the tensor via get_state_history().

METEO_VALID_RANGES: dict[str, tuple[float, float]] = {
    "temperature":          (-40.0,  60.0),
    "humidite":             (  0.0, 100.0),
    "pression":             (80000.0, 110000.0),
    "pluie":                (  0.0, 500.0),
    "pluie_intensite_max":  (  0.0, 500.0),
}

# CSV column names (output of download_toulouse_meteo.py)
# 'timestamp' is the unified name from v4+; older files use 'dd_mm_yy_hh_mm_utc'
CSV_TS_COL_OPTIONS = ["timestamp", "dd_mm_yy_hh_mm_utc"]
CSV_STATION_COL    = "station_name"
CSV_LAT_COL        = "latitude"
CSV_LON_COL        = "longitude"
CSV_ALT_COL        = "altitude"
CSV_FEATURE_COLS   = list(METEO_VALID_RANGES.keys())


@dataclass
class MeteoDataset(HybridIoTDataset):
    """
    Meteorological dataset for the HybridIoT system.

    Built from the CSV produced by download_toulouse_meteo.py.
    Inherits the full interface of HybridIoTDataset (get_data_window,
    quality_report, PyTorch DataLoader compatibility).

    The 'source' attribute records the original CSV path for traceability.
    """

    source: str = "unknown"

    # Constructor

    @classmethod
    def from_csv(
        cls,
        path: str,
        name: str                  = "",
        window_size: int           = 96,
        feature_cols: List[str]    = None,
    ) -> "MeteoDataset":
        """
        Build a MeteoDataset from the aggregated CSV of Toulouse Open Data.

        path : str
            Path to the CSV file (output of download_toulouse_meteo.py).
        name : str, optional
            Human-readable name. Defaults to the CSV filename stem.
        window_size : int
            Number of consecutive timesteps per ML training sample.
            Default 96 = 24h at 15-minute frequency.
        feature_cols : list of str, optional
            Which feature columns to load. Defaults to all known meteo features.

        Returns MeteoDataset
        """
        if not name:
            name = os.path.splitext(os.path.basename(path))[0]

        feature_cols = feature_cols or CSV_FEATURE_COLS

        # 1. Load CSV 
        with open(path, "r") as f:
            first_line = f.readline()
        sep = ";" if first_line.count(";") > first_line.count(",") else ","
        
        df = pd.read_csv(path, sep=sep)

        # Detect timestamp column name, supports both old and new format
        ts_col = None
        for candidate in CSV_TS_COL_OPTIONS:
            if candidate in df.columns:
                ts_col = candidate
                break
        if ts_col is None:
            raise ValueError(
                f"No timestamp column found in {path}. "
                f"Expected one of: {CSV_TS_COL_OPTIONS}. "
                f"Available columns: {list(df.columns)}"
            )
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)

        available_features = [c for c in feature_cols if c in df.columns]
        if not available_features:
            raise ValueError(
                f"No known feature columns found in {path}. "
                f"Expected at least one of: {feature_cols}"
            )

        # 2. Compute sampling frequency automatically from the CSV 
        # Uses the most frequent time difference between consecutive timestamps
        # of a single station. Robust to occasional gaps.
        sample_station = df[CSV_STATION_COL].iloc[0]
        station_ts = (
            df[df[CSV_STATION_COL] == sample_station][ts_col]
            .sort_values()
        )
        diffs = station_ts.diff().dropna()
        freq_minutes = int(diffs.mode()[0].total_seconds() / 60)
        print(f"  Sampling frequency detected: {freq_minutes} min")

        # 3. Build uniform timestamp grid 
        freq      = timedelta(minutes=freq_minutes)
        t_min     = df[ts_col].min().to_pydatetime()
        t_max     = df[ts_col].max().to_pydatetime()
        all_ts    = []
        t         = t_min
        while t <= t_max:
            all_ts.append(t)
            t += freq
        n_timesteps = len(all_ts)
        ts_index    = {ts: i for i, ts in enumerate(all_ts)}

        # 4. Build sensors and tensor  
        station_names = sorted(df[CSV_STATION_COL].unique())
        n_sensors     = len(station_names)
        n_types       = len(available_features)

        # Full tensor, NaN by default (missing = NaN)
        tensor = torch.full(
            (n_types, n_timesteps, n_sensors),
            float("nan"),
            dtype=torch.float32,
        )

        sensors: List[Sensor] = []

        for s_idx, station_name in enumerate(station_names):
            group = (
                df[df[CSV_STATION_COL] == station_name]
                .sort_values(ts_col)
                .reset_index(drop=True)
            )

            lat = float(group[CSV_LAT_COL].iloc[0])
            lon = float(group[CSV_LON_COL].iloc[0])
            alt = float(group[CSV_ALT_COL].iloc[0]) if CSV_ALT_COL in group.columns else 0.0

            sensor = Sensor(
                id         = str(station_name),
                position   = Position(longitude=lon, latitude=lat, altitude=alt),
                is_virtual = False,
            )

            # Fill tensor 
            for _, row in group.iterrows():
                ts = row[ts_col]
                if pd.isna(ts):
                    continue
                ts_dt = ts.to_pydatetime()
                t_idx = ts_index.get(ts_dt)
                if t_idx is None:
                    continue

                for f_idx, feat in enumerate(available_features):
                    if feat not in row or pd.isna(row[feat]):
                        continue
                    val    = float(row[feat])
                    lo, hi = METEO_VALID_RANGES.get(feat, (-9999.0, 9999.0))
                    if lo <= val <= hi:
                        tensor[f_idx, t_idx, s_idx] = val
                    # else: leave as NaN, detectable via get_state_history()

            sensors.append(sensor)

        return cls(
            sensors                = sensors,
            values                 = tensor,
            data_types             = available_features,
            start_time             = all_ts[0],
            end_time               = all_ts[-1],
            sensors_aggr_frequency = freq_minutes,
            window_size            = window_size,
            name                   = name,
            source                 = path,
        )

    # Description

    def describe(self) -> str:
        return (
            f"MeteoDataset '{self.name}'\n"
            f"  Source      : {self.source}\n"
            f"  Sensors     : {self.n_sensors} stations\n"
            f"  Data types  : {self.data_types}\n"
            f"  Tensor      : {tuple(self.values.shape)}  "
            f"(n_types × n_timesteps × n_sensors)\n"
            f"  Timesteps   : {self.n_timesteps} "
            f"(every {self.sensors_aggr_frequency} min)\n"
            f"  Time range  : {self.start_time} -> {self.end_time}\n"
            f"  NaN rate    : "
            f"{torch.isnan(self.values).float().mean().item():.2%}"
        )
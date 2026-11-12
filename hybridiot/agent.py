"""
Defines Agent: an active sensing agent in the HybridIoT system.
 
An Agent corresponds to one physical sensor measuring one type of data
(ex. only temperature, only humidity). It has its own local dataset
List[Tuple[datetime, float]] that can be initialized from a HybridIoTDataset
(base knowledge) and updated dynamically with new measurements (data stream).

Agents are situated in an Environment. To obtain data from
another agent, an agent does NOT hold a direct reference to the target:
it asks the environment via self.environment.get_data_window(...).
 
Subclass of Sensor = inherits id, position, is_virtual.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from hybridiot.environment import Environment

from .base_dataset import HybridIoTDataset
from .sensor import Sensor
from .estimator import Estimator

import numpy as np

# Valid ranges per meteorological variable 
# Values outside these ranges are considered anomalous.
METEO_VALID_RANGES: dict[str, tuple[float, float]] = {
    "temperature":          (-40.0,  60.0),
    "humidite":             (  0.0, 100.0),
    "pression":             (80000.0, 110000.0),
    "pluie":                (  0.0, 500.0),
    "pluie_intensite_max":  (  0.0, 500.0),
}

@dataclass
class Agent(Sensor):
    """
    An active sensing agent. Attributes:
    data_type : str
        The single type of data measured by this agent (e.g. "temperature").
    frequency : int
        Acquisition frequency in minutes, given a priori (a physical device
        is configured to acquire at a given frequency). Default: 15.
    dataset : List[Tuple[datetime, float]]
        Local dataset of (timestamp, value) tuples.
    fault_periods : List[Tuple[datetime, datetime]]
        Periods during which the recorded values were anomalous.
        Built and updated automatically.
    environment : Environment
        Reference to the shared environment. Set automatically when the
        agent is registered via environment.register_agent(agent).
    """
    data_type: str = "" 
    frequency: int = 15 
    dataset: List[Tuple[datetime, float]] = field(default_factory=list)  
    fault_periods: List[Tuple[datetime, datetime]] = field(default_factory=list)  
    environment: Environment = None
    local_environment: Environment = None # confidence zone: the agent's local environment
    estimated_timestamps: set = field(default_factory=set)
    last_collaborators: list = field(default_factory=list)  # Track the last collaborators used for exogenous estimation
    last_endogenous: float = field(default=None)  # Track the last endogenous estimate for comparison in exogenous estimation


    # Methods for dataset management

    def load_from_dataset(self, base_dataset: HybridIoTDataset):
        """
        Initialize the agent's dataset with a list of (timestamp, value) tuples.
        Also builds fault_periods from the loaded data automatically,
        so the agent is fully consistent after this call.
        """
        window = base_dataset.get_data_window(
            data_types=self.data_type,
            sensor_ids=self.id
        )
        values = window.valeurs.flatten().tolist()  # Convert tensor to list of values
        timestamps = window.timestamps  # List of timestamps corresponding to the values

        for ts, val in zip(timestamps, values):
            self.dataset.append((ts, val))
        self.build_fault_periods()  # Build fault periods after loading the dataset

    def add_value(self, value: float, timestamp: datetime = None, is_estimate: bool = False):
        """
        Add a new (timestamp, value) entry to the agent's dataset.
        If timestamp is not provided, use the current time.
        Also updates fault_periods automatically based on the new value.
        If is_estimate is True, the timestamp is tracked in estimated_timestamps.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        self.dataset.append((timestamp, value))
        if is_estimate:
            self.estimated_timestamps.add(timestamp)  # Track estimated timestamps
        self.update_fault_periods()  # Update fault periods after adding a new value

    def is_estimated(self, timestamp: datetime) -> bool:
        """
        Return True if the value at the given timestamp was estimated
        (added via add_value with is_estimate=True), False otherwise.
        """
        return timestamp in self.estimated_timestamps
    

    # Methods for fault detection 

    def is_fault(self, timestamp: datetime = None) -> bool:
        """
        Return True if the value at the given timestamp (or the last value
        if no timestamp is provided) is considered anomalous.
 
        Anomalous means: no value, None, NaN, or outside the valid range
        for this agent's data_type.

        """
        dataset = self.dataset
        if not dataset:
            return True  # No data means we consider it a fault

        # Find the (ts, value) entry to check
        value = None
        if timestamp:
            for ts, val in dataset:
                if ts == timestamp:
                    value = val
                    break
        else: value = dataset[-1][1]  # Get the last value if no timestamp is provided

        if value is None:
            return True  # No value found for the given timestamp means we consider it a fault
        if isinstance(value, float) and (np.isnan(value)):  # Check for NaN
            return True

        valid_range = METEO_VALID_RANGES.get(self.data_type)
        if valid_range and (value < valid_range[0] or value > valid_range[1]):
            return True  # Value outside valid range is considered a fault

        return False  # Value is valid and within range, not a fault   

    def build_fault_periods(self):
        """
        Scan the entire local dataset and build fault_periods from scratch.
        Use this AFTER load_from_dataset() to detect anomalies in the
        initial base of knowledge.

        A fault period starts at the timestamp of the first anomalous value
        and ends at the timestamp of the last consecutive anomalous value.
        """
        self.fault_periods = []   # reset before rebuilding

        previous_was_fault = False
        for ts, val in self.dataset:
            is_anomalous = self.is_fault(timestamp=ts)

            if is_anomalous and not previous_was_fault:
                # open a new fault period
                self.fault_periods.append((ts, ts))
            elif is_anomalous and previous_was_fault:
                # extend the existing fault period
                last_start = self.fault_periods[-1][0]
                self.fault_periods[-1] = (last_start, ts)

            previous_was_fault = is_anomalous
    
    def update_fault_periods(self):
        """
        Update fault_periods after a new value has been added via add_value().
        Checks only the last entry of the dataset — assumes previous entries
        have already been processed by build_fault_periods() or by previous
        calls to this method.
        Called automatically by add_value().
        """
        if not self.dataset:
            return    
        last_ts, _ = self.dataset[-1]
        if not self.is_fault(timestamp=last_ts):
            return   # last value is normal, nothing to do
        # Check if the previous value (if any) was also anomalous
        previous_was_fault = False
        if len(self.dataset) >= 2:
            prev_ts, _ = self.dataset[-2]
            previous_was_fault = self.is_fault(timestamp=prev_ts)
        if previous_was_fault and self.fault_periods:
            # extend the last fault period
            last_start = self.fault_periods[-1][0]
            self.fault_periods[-1] = (last_start, last_ts)
        else:
            # open a new fault period
            self.fault_periods.append((last_ts, last_ts))


    # Methods for data access

    def slice(self, start_time: datetime = None, end_time: datetime = None) -> List[Tuple[datetime, float]]:
        """
        Return a list of (timestamp, value) tuples from the dataset that fall within the
        specified time range [start_time, end_time]. None on either side means no bound.
        If no data falls within the range, returns an empty list.
        """
        if not self.dataset:
            return []
        if start_time is None:
            start_time = min(entry[0] for entry in self.dataset)
        if end_time is None:
            end_time = max(entry[0] for entry in self.dataset)
        return [entry for entry in self.dataset if start_time <= entry[0] <= end_time]


    # Methods for environment interaction: the agent never comunicates directly with another agent.

    def request_data_window(
        self,
        start_time: datetime = None,
        end_time:   datetime = None,
    ) -> Dict:
        """
        Request data windows from collaborating agents THROUGH the environment.
 
        The agent does NOT specify who to ask — the Environment decides
        which agents are pertinent (via get_neighbours) and returns
        their data windows.
 
        Returns
        dict {agent_id: DataWindow}
            Data windows from each collaborator selected by the Environment.
            Empty dict if the agent is not registered in an environment.
        """
        if self.environment is None:
            return {}
        return self.environment.ask_data_window(
            requesting_agent = self,
            start_time       = start_time,
            end_time         = end_time,
        )

    # ESTIMATION METHODS 
    # The estimation methods are LOCAL to the agent.
    # The Agent prepares the data, the Estimator does the calculation.

    # Utility methods for data preparation: _get_valid_values, _build_candidate_windows, _get_valid_entries, _build_window_ending_at.

    def _get_valid_values(self, before: datetime = None) -> List[float]:
        """
        Extract valid (non-NaN, non-None) values from the local dataset,
        optionally only those with timestamp strictly before `before`.

        Returns a list of floats in chronological order.
        """
        values = []
        for ts, val in self.dataset:
            if before is not None and ts >= before:
                break
            if self._is_valid_value(val):  # Use the same criteria as is_fault() but without scanning the dataset again
                values.append(val)
        return values
    
    def _build_candidate_windows(self, values: List[float]) -> List[List[float]]:
        """
        Split a list of values into consecutive, non-overlapping windows
        of size Estimator.WINDOW_SIZE.
        Remaining values that don't fill a complete window are discarded.

        values : List[float]
            Chronologically ordered values from which to extract windows.

        Returns
        List[List[float]]
            List of complete ACWs (each of length WINDOW_SIZE).
        """
        size = Estimator.WINDOW_SIZE
        windows = []
        for i in range(0, len(values) - size + 1, size):
            windows.append(values[i : i + size])
        return windows
    
    # _get_valid_entries is a (timestamp, value) variant of _get_valid_values, useful for exogenous estimation where we need the timestamps to build Sigma. 
    def _get_valid_entries(
        self, before: datetime = None
    ) -> List[Tuple[datetime, float]]:
        """
        Extract valid (timestamp, value) entries from the local dataset,
        optionally only those with timestamp strictly before `before`.
        Same filtering as _get_valid_values, but keeps the timestamps 
        needed to build the set of similar instants Sigma.
        """
        entries = []
        for ts, val in self.dataset:
            if before is not None and ts >= before:
                break
            if self._is_valid_value(val):  # Use the same criteria as is_fault() but without scanning the dataset again
                entries.append((ts, val))
        return entries
    
    def _build_window_ending_at(
        self,
        entries: List[Tuple[datetime, float]],
        end_ts:  datetime,
    ) -> Optional[List[float]]:
        """
        Build an ACW of WINDOW_SIZE values ending at (or immediately
        before) end_ts, from the given valid entries.

        Returns None if fewer than WINDOW_SIZE values are available
        at that instant.
        """
        size = Estimator.WINDOW_SIZE
        past = [val for ts, val in entries if ts <= end_ts]
        if len(past) < size:
            return None
        return past[-size:]

    def _is_valid_value(self, val) -> bool:
        """
        True if the value is usable for estimation: not None, not NaN,
        and within the valid range for this agent's data_type.
        Out-of-range values (e.g. the -50 sentinel used as a missing-data
        marker in the Toulouse OpenData CSV) would otherwise poison the
        ACW distances and transitions.

        Note: same criteria as is_fault(), but takes the value directly
        instead of searching the dataset by timestamp, is_fault() does a
        linear scan, which would make the extractors O(n^2).
        """
        if val is None:
            return False
        if isinstance(val, float) and np.isnan(val):
            return False
        valid_range = METEO_VALID_RANGES.get(self.data_type)
        if valid_range and (val < valid_range[0] or val > valid_range[1]):
            return False
        return True

    
    # Method 1: estimation from history uses self.dataset only

    def estimate_from_history(self, timestamp: datetime) -> float | None:
        """
        Method 1: estimation from history (endogenous, non-cooperative).
        Prepares the data from the local dataset, then delegates the
        calculation to Estimator.estimate_from_history().
        Steps:
          1. Extract all valid values before `timestamp`
          2. Build current_window: last WINDOW_SIZE values
          3. Build candidate_windows: non-overlapping windows from
             the remaining history
          4. Call Estimator.estimate_from_history()
        timestamp : datetime
            The timestamp of the missing value to estimate.

        Returns
        float
            The estimated value.
        None
            If there is not enough history to estimate
            (less than 2 * WINDOW_SIZE valid values).
        """
        size = Estimator.WINDOW_SIZE

        # 1. Get all valid values before the target timestamp
        valid_values = self._get_valid_values(before=timestamp)

        # 2. We need at least WINDOW_SIZE for current_window
        #    + WINDOW_SIZE for at least one candidate
        if len(valid_values) < 2 * size:
            return None

        # 3. current_window = last WINDOW_SIZE values
        current_window = valid_values[-size:]
        last_value = current_window[-1]

        # 4. candidate_windows from the remaining history
        history = valid_values[:-size]
        candidate_windows = self._build_candidate_windows(history)

        if not candidate_windows:
            return None

        # 5. Delegate calculation to Estimator
        return Estimator.estimate_from_history(
            current_window=current_window,
            candidate_windows=candidate_windows,
            last_value=last_value,
        )
    
    # Method 3: exogenous cooperative estimation
 
    def estimate_exogenous(self, timestamp: datetime, collab_ratio = 0.3) -> Optional[float]:
        """
        Method 3: exogenous cooperative estimation (heterogeneous collaborators).
        Steps:
          1. Compute the endogenous estimate E_end from own history,
             and identify Sigma: the instants associated with the ACWs
             most similar to the current window.
          2. Send Sigma to the Environment (ask_weights), which selects
             heterogeneous collaborators; each returns a weight
             computed on ITS OWN data at the instants Sigma.
          3. Combine E_end and the received weights via the histogram
             (Estimator.combine_estimates).
 
        Returns
        float
            The exogenous estimate. Falls back to the endogenous
            estimate if no collaborator contributed.
        None
            If there is not enough history for the endogenous step
            (the exogenous method CONTAINS the endogenous one).
        """
        size = Estimator.WINDOW_SIZE
 
        # 1. Endogenous step on own history 
        entries = self._get_valid_entries(before=timestamp)
        if len(entries) < 2 * size:
            return None
 
        values = [val for _, val in entries]
        current_window = values[-size:]
        last_value = current_window[-1]
 
        history = values[:-size]
        candidate_windows = self._build_candidate_windows(history)
        if not candidate_windows:
            return None
 
        endogenous = Estimator.estimate_from_history(
            current_window=current_window,
            candidate_windows=candidate_windows,
            last_value=last_value,
        )
 
        # Sigma: instants of the most similar ACWs (timestamp of the last
        # value of each selected window). Candidate windows are
        # consecutive non-overlapping blocks of `size` values, so
        # window i covers entries [i*size, (i+1)*size) of the history.
        ranked = Estimator.rank_by_similarity(current_window, candidate_windows)
        sigma = [
            entries[(idx + 1) * size - 1][0]   # ts of the window's last value
            for _, idx in ranked[:10]
        ]
 
        # 2. Cooperative step through the Environment 
        if self.environment is None:
            return endogenous
 
        weights = self.environment.ask_weights(
            requesting_agent=self,
            sigma=sigma,
            timestamp=timestamp,
            collab_ratio=collab_ratio
        )
 
        if not weights:
            return endogenous

        # Track the last collaborators used for exogenous estimation and last endogenous estimate
        self.last_collaborators = list(weights.keys()) if weights else [] 
        self.last_endogenous = endogenous  
 
        # 3. Histogram combination 
        return Estimator.combine_estimates(
            endogenous_estimate=endogenous,
            cooperative_weights=list(weights.values()),
            last_value=last_value,
        )
    
    # Collaborator side
 
    def compute_cooperative_weight(
        self,
        sigma:     List[datetime],
        timestamp: datetime,
    ) -> Optional[float]:
        """
        Collaborator-side computation for the exogenous method.
 
        Called (via the Environment) when another agent asks for help.
        This agent compares its OWN current ACW at `timestamp` with its
        OWN ACWs at the instants Sigma, and returns a weight: a variation,
        not a value, which is what makes heterogeneous cooperation
        possible (variations are dimensionless across data types).
 
        Returns
        float
            The weight computed with Estimator.compute_weight().
        None
            If this agent cannot contribute (not enough data at
            `timestamp` or at the instants Sigma).
        """
        entries = self._get_valid_entries(before=timestamp)
 
        current_window = self._build_window_ending_at(
            entries, end_ts=entries[-1][0]
        ) if entries else None
        if current_window is None:
            return None
 
        windows_at_sigma = []
        for instant in sigma:
            window = self._build_window_ending_at(entries, end_ts=instant)
            if window is not None:
                windows_at_sigma.append(window)
 
        if not windows_at_sigma:
            return None
 
        return Estimator.compute_weight(
            current_window=current_window,
            candidate_windows=windows_at_sigma,
            k=len(windows_at_sigma),
        )

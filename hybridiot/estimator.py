"""
Defines Estimator: pure calculation helper for estimating missing sensor values.

Holds NO reference to Agent or Environment.
Receives already-prepared data (ACWs as List[float]) and returns a float.

Window size is fixed at WINDOW_SIZE = 8 for this phase.
Variable-size windows (thesis Section 5.10) will be integrated
once the acquisition frequency mechanism is implemented.
"""

from typing import List, Optional, Tuple


class Estimator:
    """
    Pure calculation helper.

    Responsibilities:
      - compute the distance between two ACWs
      - select the most similar ACWs
      - compute the weight w_t and return the estimate

    NOT its responsibilities (belong to Agent):
      - access the local dataset
      - build ACWs from the dataset
      - communicate with the Environment
    """

    WINDOW_SIZE: int = 8  # fixed 

    # Distance between two ACWs

    @staticmethod
    def distance(window_a: List[float], window_b: List[float]) -> float:
        """
        Distance between two ACWs of the same size
            d(a, b) = (1/n) * Σ |a[i] - b[i]|

        window_a, window_b : List[float]
            Two ACWs of the same length.

        Returns 
        float 
            distance >= 0 (0 = identical windows).
        """
        n = len(window_a)
        if n == 0:
            return 0.0
        return sum(abs(a - b) for a, b in zip(window_a, window_b)) / n
    
    # Similarity ranking for exogenous estimation
 
    @staticmethod
    def rank_by_similarity(
        current_window:    List[float],
        candidate_windows: List[List[float]],
    ) -> List[Tuple[float, int]]:
        """
        Rank candidate ACWs by similarity to the current window.
 
        Returns a list of (distance, index) pairs sorted by increasing distance. 
        The index refers to the position of the candidate in candidate_windows:
        the caller (Agent) can map it back to timestamps, to build the set of similar instants.
        """
        scored = [
            (Estimator.distance(current_window, candidate), i)
            for i, candidate in enumerate(candidate_windows)
        ]
        scored.sort(key=lambda x: x[0])
        return scored
    
    # Weight computation for any estimation method

    @staticmethod
    def compute_weight(
        current_window:    List[float],
        candidate_windows: List[List[float]],
        k:                 int = 10,
    ) -> Optional[float]:
        """
        Weight w_t: weighted average of the final transitions of the k ACWs most similar to current_window.
 
        Used in two contexts:
          - by the agent itself on its own history (method 1)
          - by each collaborator on its own ACWs at the instants Sigma
            (method 3, exogenous)
 
        Returns
        float
            The weight w_t.
        None
            If no candidate is usable (empty list).
        """
        if not candidate_windows:
            return None
 
        ranked = Estimator.rank_by_similarity(current_window, candidate_windows)

        # ranked is sorted by increasing distance so the last one is the max
        d_max = ranked[-1][0]
        if d_max == 0.0:
            return None   # every candidate is identical to the current window

        normalized = [(d / d_max, idx) for d, idx in ranked]
        selected = [(d, idx) for d, idx in normalized
                    if d != 0.0 and d != 1.0][:k]
        if not selected:
            return None
 
        numerator = 0.0
        denominator = 0.0
 
        for d_norm, idx in selected:
            similarity = 1.0 - d_norm
            candidate = candidate_windows[idx]
            numerator += similarity * (candidate[-1] - candidate[-2])
            denominator += similarity

        if abs(denominator) < 1e-3:
            return None

        return numerator / denominator



    # Method 1: estimation from history

    @staticmethod
    def estimate_from_history(
        current_window:    List[float],
        candidate_windows: List[List[float]],
        last_value:        float,
        k:                 int = 10,
    ) -> float:
        """
        Method 1: estimation from history (endogenous, non-cooperative).

        E_t = last_value + w_t 
 
        Returns
        float
            the estimated value E_t.
            If no candidate is available or usable, returns last_value.
        """
        w_t = Estimator.compute_weight(current_window, candidate_windows, k)
        if w_t is None:
            return last_value
        return last_value + w_t
    

    # Method 3: exogenous combination (histogram-based, cooperative)

    @staticmethod
    def combine_estimates(
        endogenous_estimate: float,
        cooperative_weights: List[float],
        last_value:          float,
        n_bins:              int = 10,
    ) -> float:
        """
        Method 3: combine the endogenous estimate with the weights
        received from heterogeneous collaborators.
 
        Steps:
          1. Each cooperative weight w yields a candidate estimate
             last_value + w. The endogenous estimate is included in
             the set (paper: the histogram contains all estimates).
          2. Build a histogram H of the candidate estimates (n_bins).
          3. Select the bin whose mean is closest to the endogenous
             estimate:  f(H_k) = |mean(H_k) - E_end|.
             This discards misleading collaborators, whose estimates
             fall in bins far from E_end.
          4. Return E_exo = (mean(best_bin) + E_end) / 2 — an equal
             weighting of the cooperative result and the endogenous
             estimate.
 
        endogenous_estimate : float
            E_end, computed by estimate_from_history().
        cooperative_weights : List[float]
            Weights received from the collaborators (one per agent).
        last_value : float
            Last perceived value E_{t-1} of the requesting agent.
        n_bins : int
            Number of histogram bins (paper: 10, Freedman-Diaconis
            median over all sensors).
 
        Returns
        float
            The exogenous estimate E_exo. If no cooperative weight is
            available, returns the endogenous estimate unchanged.
        """
        if not cooperative_weights:
            return endogenous_estimate
 
        # 1. Candidate estimates
        estimates = [last_value + w for w in cooperative_weights]
 
        # 2. Histogram
        lo, hi = min(estimates), max(estimates)
        if hi == lo:
            # All estimates identical: nothing to filter
            return (lo + endogenous_estimate) / 2.0
 
        width = (hi - lo) / n_bins
        bins: List[List[float]] = [[] for _ in range(n_bins)]
        for e in estimates:
            idx = min(int((e - lo) / width), n_bins - 1)
            bins[idx].append(e)
 
        # 3. Bin whose mean is closest to the endogenous estimate
        best_mean: Optional[float] = None
        best_diff = float("inf")
        for b in bins:
            if not b:
                continue
            mean = sum(b) / len(b)
            diff = abs(mean - endogenous_estimate)
            if diff < best_diff:
                best_diff = diff
                best_mean = mean
 
        # 4. Equal weighting of cooperative result and endogenous estimate
        return (best_mean + endogenous_estimate) / 2.0
 

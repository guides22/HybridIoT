"""
Defines Environment: the shared environment of the HybridIoT multi-agent system.

In a multi-agent system, agents do not interact directly with each other:
they are situated in an environment, where they can act and exchange
information with the other agents. The Environment mediates all exchanges.

Multiple Environment instances can coexist: a global environment shared
by all agents, and (in the future) local environments representing what
each agent can perceive (confidence zones). Instances holding references
to the same Agent objects share their state: an update to an agent's
dataset is visible from every environment that references it.

Responsibilities:
- hold the list of registered agents
- hold the base of knowledge (HybridIoTDataset)
- mediate data window requests between agents via get_data_window()
"""

from datetime import datetime
from typing import Dict, List, Optional

import torch
import random
import math

from .base_dataset import HybridIoTDataset
from .data_window import DataWindow


class Environment:
    """
    Shared environment of the HybridIoT multi-agent system.

    agents : List[Agent]
        All agents registered in the environment.
    base_dataset : HybridIoTDataset
        The base of knowledge shared by the system.
    """

    def __init__(self, base_dataset: HybridIoTDataset = None):
        self.agents:       List = []
        self.base_dataset: HybridIoTDataset = base_dataset

    # Agent management 

    def register_agent(self, agent) -> None:
        """
        Register an agent in the environment.
        Sets the agent's reference to this environment.
        """
        if agent not in self.agents:
            self.agents.append(agent)
        # register_agent claims the agent: it is meant for the GLOBAL environment only. 
        # To populate a local environment (confidence zone) without overwriting 
        # agent.environment, use add_agent().
        agent.environment = self

    def add_agent(self, agent) -> None:
        """
        Add an agent to this environment BY REFERENCE, without claiming
        it: unlike register_agent(), this does NOT set agent.environment.
        Used to populate local environments (confidence zones): the agent
        keeps its reference to the global environment, and is simply
        perceivable within this local one.
        """
        if agent not in self.agents:
            self.agents.append(agent)

    def get_agent(self, agent_id: str, data_type: str = None):
        """
        Return the registered agent with the given id (and optionally
        the given data_type, since one station can have several agents).
        Returns None if not found.
        """
        for agent in self.agents:
            if agent.id == agent_id:
                if data_type is None or agent.data_type == data_type:
                    return agent
        return None

    # get agents by type, name or other criteria can be added here as needed

    def get_agents_by_type(self, data_type: str) -> List:
        """Return all registered agents measuring the given data type."""
        return [a for a in self.agents if a.data_type == data_type]


    # Collaboration selection
 
    def get_neighbours(self, requesting_agent, same_data_type: bool = True, collab_ratio: float = 1.0) -> List:
        """
        Return the agents that are pertinent and accessible for the
        requesting agent, using a random selection weighted by geographic
        proximity: closer agents have a higher probability of being selected.
 
        Current logic (simple, to be refined):
          1. filter agents by same_data_type, excluding the requester
          2. compute geographic distance to each candidate
          3. assign weights inversely proportional to distance
          4. randomly select a subset based on those weights removing each chosen agent from the pool to avoid duplicates

        same_data_type: bool
            True -> candidates measure the SAME data type (homogeneous, used by ask_data_window / confidence zone logic)
            False -> candidates measure a DIFFERENT data type (heterogeneous, used by the exogenous method)
 
        Returns a List[Agent]: selected collaborators (may vary between calls due to randomness)
        """
        # 1. filter candidates: same data_type, exclude self
        candidates = [
            a for a in self.agents
            if a.id != requesting_agent.id
            and (a.data_type == requesting_agent.data_type) == same_data_type
            and len(a.dataset) > 0    # exclude agents with no data
        ]
        if not candidates:
            return []
 
        # 2. compute distances (Euclidean on lat/lon)
        distances = []
        for a in candidates:
            dlat = a.position.latitude  - requesting_agent.position.latitude
            dlon = a.position.longitude - requesting_agent.position.longitude
            dist = math.sqrt(dlat**2 + dlon**2)
            dist = max(dist, 0.0001)   # avoid division by zero
            distances.append(dist)
 
        # 3. weights = 1/distance (closer = higher weight)
        weights = [1.0 / d for d in distances]
 
        # 4. select a subset of three candidates (3) based on weights without duplicates
        k = max(1, int(collab_ratio * len(candidates)))
        pool = list(candidates)
        pool_weights = list(weights)
        selected = []

        for _ in range(k):
            if not pool:
                break
            chosen = random.choices(pool, weights=pool_weights, k=1)[0]
            selected.append(chosen)
            idx = pool.index(chosen)
            pool.pop(idx)
            pool_weights.pop(idx)

        return selected


    # Mediated data access 

    def ask_data_window(
        self,
        requesting_agent,
        start_time: datetime = None,
        end_time:   datetime = None,
    ) -> Dict[str, DataWindow]:
        """
        The requesting agent asks the Environment for data windows.
        The Environment:
          1. selects the pertinent collaborators via get_neighbours()
          2. retrieves their data windows
          3. returns them to the requester
        The agent does NOT specify who to ask, the Environment decides.
 
        Returns a dict {agent_id: DataWindow}: data windows from each selected collaborator.
        """
        neighbours = self.get_neighbours(requesting_agent)
        result = {}
 
        for neighbour in neighbours:
            entries = neighbour.slice(start_time, end_time)
 
            if not entries:
                result[neighbour.id] = DataWindow(
                    type         = neighbour.data_type,
                    valeurs      = torch.empty((1, 0, 1), dtype=torch.float32),
                    interv_temps = (start_time, end_time),
                    sensor_ids   = [neighbour.id],
                    timestamps   = [],
                )
            else:
                timestamps = [ts  for ts, _ in entries]
                values     = [val for _, val in entries]
                tensor     = torch.tensor(values, dtype=torch.float32).reshape(1, -1, 1)
 
                result[neighbour.id] = DataWindow(
                    type         = neighbour.data_type,
                    valeurs      = tensor,
                    interv_temps = (timestamps[0], timestamps[-1]),
                    sensor_ids   = [neighbour.id],
                    timestamps   = timestamps,
                )
 
        return result
    
    def ask_weights(
        self,
        requesting_agent,
        sigma:     List[datetime],
        timestamp: datetime,
        collab_ratio: float = 0.3
    ) -> Dict[str, float]:
        """
        Communication method for the exogenous estimation.
        The requesting agent provides the instants associated with
        the ACWs most similar to its current window. The Environment:
          1. selects HETEROGENEOUS collaborators via get_neighbours()
          2. asks each one to compute a weight on ITS OWN data
          3. returns the weights to the requester

        Agents never exchange raw values here, only weights
        (variations), which is what makes heterogeneous cooperation
        possible.

        Returns a dict {agent_id: weight}.
        """
        neighbours = self.get_neighbours(requesting_agent, same_data_type=False, collab_ratio=collab_ratio)
        result = {}

        for neighbour in neighbours:
            weight = neighbour.compute_cooperative_weight(sigma, timestamp)
            if weight is not None:
                result[f"{neighbour.id} [{neighbour.data_type}]"] = weight
        return result


    # Utilities

    def __repr__(self) -> str:
        ds_name = self.base_dataset.name if self.base_dataset else "None"
        return (
            f"Environment(agents={len(self.agents)}, "
            f"base_dataset={ds_name!r})"
        )
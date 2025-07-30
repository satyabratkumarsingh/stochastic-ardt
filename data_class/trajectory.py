from dataclasses import dataclass
from typing import Any, List, Optional

@dataclass
class Trajectory:
    episode_id: int
    obs: List[Any]
    actions: List[Any]
    rewards: List[float]
    adv_actions: List[Any]
    adv_rewards: List[float]
    infos: List[dict]
    dones: List[bool]
    minimax_returns_to_go: Optional[List[float]] = None
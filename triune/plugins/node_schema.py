"""Node-based Visual Pipeline Schema.

Defines serializable Node, Port, and Graph Data Transfer Objects for Triune Studio GUI
and CLI pipeline execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class Port:
    name: str
    type: str  # "dataset", "model", "optimizer", "loss", "metric", "tensor", "text"
    description: str = ""


@dataclass
class NodeSchema:
    id: str
    name: str
    category: str  # "data", "model", "optimizer", "trainer", "evaluation", "visualization", "custom"
    inputs: List[Port] = field(default_factory=list)
    outputs: List[Port] = field(default_factory=list)
    config_params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

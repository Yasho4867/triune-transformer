"""Dynamic Plugin & Node Extension Registry with @register_node decorator."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Type


class NodeRegistry:
    """Centralized Registry for Framework Nodes, Custom Layers, and Loss Functions."""

    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: Optional[str] = None,
        category: str = "Custom",
        description: str = "",
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
    ) -> Callable:
        """Decorator to register any Python function or class into the Triune ecosystem."""

        def decorator(target: Any) -> Any:
            node_name = name or (target.__name__ if hasattr(target, "__name__") else str(target))
            sig = inspect.signature(target) if inspect.isfunction(target) or inspect.isclass(target) else None

            params = {}
            if sig:
                for param_name, param in sig.parameters.items():
                    if param_name in ("self", "cls"):
                        continue
                    params[param_name] = {
                        "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                        "default": param.default if param.default != inspect.Parameter.empty else None
                    }

            schema = {
                "name": node_name,
                "category": category,
                "description": description or (target.__doc__ or ""),
                "inputs": inputs or ["in"],
                "outputs": outputs or ["out"],
                "parameters": params,
                "target": target,
            }

            self._nodes[node_name] = schema
            try:
                target._triune_node_schema = schema
            except Exception:
                pass
            return target

        return decorator

    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        return self._nodes.get(name)

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        return self.get_node(name)

    def list_nodes(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": meta["name"],
                "category": meta["category"],
                "description": meta["description"],
                "inputs": meta["inputs"],
                "outputs": meta["outputs"],
                "parameters": meta["parameters"],
            }
            for meta in self._nodes.values()
        ]


# Global Node Registry Instances & Aliases
global_registry = NodeRegistry()
node_registry = global_registry
register_node = global_registry.register
NODE_REGISTRY = global_registry._nodes


def list_registered_nodes() -> List[Dict[str, Any]]:
    return global_registry.list_nodes()


def get_node_executor(name: str) -> Optional[Any]:
    node = global_registry.get_node(name)
    return node["target"] if node else None

from __future__ import annotations
from typing import Any

class DefaultSimulator:
    def __init__(self, **kwargs: Any) -> None:
        raise ImportError(
            "RuntimeSimulator.init: Missing or Incorrect wiring for your simulator driver. Correctly install your simulator runtime"
            " components for your test domain and then wire the simulator with the correct simulator driver in this directory."
            " Also refer to this simulator driver from your configuration file."
        )

    def observe(self, agent_id: str, component_id: str, params: dict[str, Any]) -> Any:
        raise ImportError(
            "RuntimeSimulator.observe: Missing or Incorrect wiring for your simulator driver. Correctly install your simulator runtime"
            " components for your test domain and then wire the simulator with the correct simulator driver in this directory."
            " Also refer to this simulator driver from your configuration file."
        )

    def operate(
        self,
        agent_id: str,
        component_id: str,
        action: Any,
        params: dict[str, Any],
        stats: Any = None,
    ) -> Any:
        raise ImportError(
            "RuntimeSimulator.operate: Missing or Incorrect wiring for your simulator driver. Correctly install your simulator runtime"
            " components for your test domain and then wire the simulator with the correct simulator driver in this directory."
            " Also refer to this simulator driver from your configuration file."
        )

    def reset(self, context: Any = None) -> None:
        raise ImportError(
            "RuntimeSimulator.reset: Missing or Incorrect wiring for your simulator driver. Correctly install your simulator runtime"
            " components for your test domain and then wire the simulator with the correct simulator driver in this directory."
            " Also refer to this simulator driver from your configuration file."
        )

    def initialize_stats(self, agent_id: str, params: dict[str, Any]) -> Any:
        raise ImportError(
            "RuntimeSimulator.initialize_stats: Missing or Incorrect wiring for your simulator driver. Correctly install your simulator runtime"
            " components for your test domain and then wire the simulator with the correct simulator driver in this directory."
            " Also refer to this simulator driver from your configuration file."
        )


import typing
if typing.TYPE_CHECKING:
    from runtime.factory import RuntimeSimulator
_istype_DefaultSimulator: typing.Type[RuntimeSimulator] = DefaultSimulator

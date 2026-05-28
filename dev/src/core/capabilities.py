from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    return value


@dataclass(frozen=True)
class CapabilityProfile(Mapping[str, Any]):
    _items: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "CapabilityProfile":
        if not data:
            return cls()
        return cls(
            tuple(sorted((str(key), _freeze(value)) for key, value in data.items()))
        )

    @classmethod
    def from_legacy_tags(
        cls,
        tags: frozenset[str] | list[str] | tuple[str, ...],
    ) -> "CapabilityProfile":
        return cls.from_mapping({str(tag): True for tag in tags})

    def __getitem__(self, key: str) -> Any:
        return dict(self._items)[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._items)

    def key(self) -> str:
        return capability_key(self)


def capability_key(capabilities: Mapping[str, Any] | frozenset[str] | list[str]) -> str:
    if isinstance(capabilities, Mapping):
        return ",".join(
            f"{key}={repr(value)}" for key, value in sorted(capabilities.items())
        )
    return ",".join(sorted(str(item) for item in capabilities))

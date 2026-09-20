"""String-keyed registry for pluggable components."""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterable, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Map string keys to classes."""

    def __init__(self, name: str) -> None:
        """Create named registry."""
        self._name = name
        self._entries: Dict[str, Type[T]] = {}

    def register(self, key: str) -> Callable[[Type[T]], Type[T]]:
        """Register class under key."""

        def _wrap(cls: Type[T]) -> Type[T]:
            norm = key.lower()
            if norm in self._entries:
                raise KeyError(
                    f"{self._name} registry already has a '{key}' entry "
                    f"({self._entries[norm].__name__})"
                )
            self._entries[norm] = cls
            return cls

        return _wrap

    def get(self, key: str) -> Type[T]:
        """Look up class by key."""
        norm = key.lower()
        if norm not in self._entries:
            raise KeyError(
                f"unknown {self._name} '{key}'. registered: {sorted(self._entries)}"
            )
        return self._entries[norm]

    def build(self, key: str, *args, **kwargs) -> T:
        """Build instance by key."""
        return self.get(key)(*args, **kwargs)

    def keys(self) -> Iterable[str]:
        """Return registered keys."""
        return self._entries.keys()

    def __contains__(self, key: str) -> bool:
        """Check if key is registered."""
        return key.lower() in self._entries


BACKBONES: "Registry" = Registry("backbone")


def register_backbone(key: str):
    """Register backbone adapter class."""
    return BACKBONES.register(key)


def get_backbone(key: str):
    """Look up backbone class by name."""
    return BACKBONES.get(key)


def list_backbones() -> list:
    """List registered backbone names."""
    return sorted(BACKBONES.keys())

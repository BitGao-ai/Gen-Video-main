"""A tiny, type-safe registry used for pluggable backbones and components.

The registry is the mechanism behind *backbone compatibility* (user requirement
#2): a new backbone is added by writing an adapter and decorating it with
``@register_backbone("name")`` — no other file changes. Construction is decoupled
from the call site, which depends only on the string key in the config.
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterable, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps string keys to classes (or factories) of a common base type."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: Dict[str, Type[T]] = {}

    def register(self, key: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator: register a class under ``key`` (case-insensitive)."""

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
        norm = key.lower()
        if norm not in self._entries:
            raise KeyError(
                f"unknown {self._name} '{key}'. registered: {sorted(self._entries)}"
            )
        return self._entries[norm]

    def build(self, key: str, *args, **kwargs) -> T:
        return self.get(key)(*args, **kwargs)

    def keys(self) -> Iterable[str]:
        return self._entries.keys()

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._entries


# The single backbone registry instance shared across the framework.
# Concrete adapters import this and decorate themselves.
BACKBONES: "Registry" = Registry("backbone")


def register_backbone(key: str):
    return BACKBONES.register(key)

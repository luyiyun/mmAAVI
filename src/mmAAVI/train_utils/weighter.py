from typing import Optional, Sequence, Any, Union


WEIGHT = Union[float, Sequence[float]]


class Weighter:

    def __init__(self) -> None:
        self._all: list[dict[str, Any]] = []
        self._key_weights = {}
        self._prefix_weights = {}
        self._select_weights = {}
        self._else_weight = None

    def add_weight(
        self,
        value: WEIGHT,
        interval: int = 1,
        key: Optional[str] = None,
        prefix: Optional[str] = None,
        selection: Optional[Sequence[str]] = None,
        else_weight: bool = False,
    ) -> None:
        assert (
            (key is None)
            + (prefix is None)
            + (selection is None)
            + else_weight
        ) == 1

        elem = {"interval": interval, "index": 0}
        if isinstance(value, float):
            elem["value"] = value
            elem["is_const"] = True
        else:
            elem["func"] = value
            elem["is_const"] = False

        if key is not None:
            self._key_weights[key] = elem
        if prefix is not None:
            self._prefix_weights[key] = elem
        if selection is not None:
            self._select_weights[tuple(selection)] = elem
        if else_weight:
            self._else_weight = elem

    def value(
        self,
        key: Optional[str] = None,
        key_for_prefix: Optional[str] = None,
        key_for_select: Optional[Sequence[str]] = None,
        else_weight: bool = False,
    ) -> float:
        assert (
            (key is None)
            + (key_for_prefix is None)
            + (key_for_select is None)
            + else_weight
        ) == 1

        if else_weight:
            elem = self._else_weight
        elif key is not None:
            elem = self._key_weights[key]
        elif key_for_prefix is not None:
            for prefix, elem in self._prefix_weights.items():
                if key_for_prefix.startswith(prefix):
                    break
        elif key_for_select is not None:
            for selections, elem in self._select_weights.items():
                if key_for_select in selections:
                    break

        if elem["is_const"]:
            return elem["value"]

        valuei = elem["func"](elem["index"])
        elem["index"] += elem["interval"]
        return valuei

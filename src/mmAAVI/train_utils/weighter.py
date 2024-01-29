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
        self, value: WEIGHT, interval: int = 1,
        key: Optional[str] = None,
        prefix: Optional[str] = None,
        selection: Optional[Sequence[str]] = None,
        else_weight: bool = False,
    ) -> None:
        assert ((key is None) + (prefix is None) +
                (selection is None) + else_weight) == 1

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
        self, key: Optional[str] = None,
        key_for_prefix: Optional[str] = None,
        key_for_select: Optional[Sequence[str]] = None,
        else_weight: bool = False,
    ) -> float:
        assert ((key is None) +
                (key_for_prefix is None) +
                (key_for_select is None) + else_weight) == 1

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


# class Weighter:
#
#     def __init__(self, nepoches: int, **params: Dict[str, Dict]) -> None:
#         params = deepcopy(params)
#         self.nepoches = nepoches
#         self.t = np.arange(nepoches) / (nepoches - 1)
#         self.weightes = {}
#         for k, v in params.items():
#             # dictconfig对象不支持pop
#             if isinstance(v, DictConfig):
#                 v = OmegaConf.to_object(v)
#             funcname = v.pop("func", "constant")
#             if funcname == "constant":
#                 self.weightes[k] = np.full(nepoches, v["value"])
#             elif funcname == "step":
#                 assert all([s in v for s in ["start", "step", "multiply"]])
#                 # 保证顺序
#                 ind = np.argsort(v["steps"])
#                 steps = np.array(v["steps"])[ind]
#                 res = np.full(nepoches, fill_value=v["start"])
#                 for si in steps:
#                     res[self.t >= si] *= v["multiply"]
#                 self.weightes[k] = res
#             elif funcname == "ladder":
#                 assert all([s in v for s in ["x", "y"]])
#                 x, y = v["x"], v["y"]
#                 assert len(x) == len(y)
#                 n = len(x)
#                 res = np.zeros_like(self.t)
#                 for i, j in zip(range(n - 1), range(1, n)):
#                     res[(self.t >= x[i]) & (self.t < x[j])] = y[i]
#                 res[(self.t >= x[-1])] = y[-1]
#                 self.weightes[k] = res
#             elif funcname == "neg_exp":
#                 assert all([s in v for s in ["max", "min", "alpha", "beta"]])
#                 self.weightes[k] = np.maximum(
#                     v["max"] * (1 + v["alpha"] * self.t) ** (-v["beta"]),
#                     v["min"]
#                 )
#             elif funcname == "exp_frac":
#                 assert all([s in v for s in ["max", "min", "delta"]])
#                 x = np.exp(-v["delta"] * self.t)
#                 self.weightes[k] = np.maximum(
#                     v["max"] * (1 - x) / (1 + x), v["min"]
#                 )
#             else:
#                 raise NotImplementedError
#
#     def __contains__(self, item: str) -> bool:
#         return item in self.weightes
#
#     def at(self, k: str, e: int) -> float:
#         return self.weightes[k][e]
#
#     def log(self, writer: Optional[SummaryWriter]) -> None:
#         for k, arr in self.weightes.items():
#             for i in range(self.nepoches):
#                 writer.add_scalar("weight/%s" % k, arr[i], i)
#         writer.flush()
#
#     def plot(self):
#         fig, ax = plt.subplots(figsize=(5, 5))
#         x = np.arange(self.nepoches)
#         for k, y in self.Wlambdas.items():
#             ax.plot(x, y, label=k)
#         fig.legend()
#         return fig, ax

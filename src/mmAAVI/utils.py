import inspect
import json
import os
import pickle
import random
from functools import wraps
from typing import Sequence

import numpy as np
import torch


def read_pkl(fn):
    with open(fn, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_pkl(obj, fn):
    with open(fn, "wb") as f:
        pickle.dump(obj, f)


def read_json(fn):
    with open(fn, "r") as f:
        obj = json.load(f)
    return obj


def save_json(obj, fn):
    with open(fn, "w") as f:
        json.dump(obj, f)


def setup_seed(seed: int, deterministic: bool = False) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def save_args(exclude: Sequence[str] = ()):
    def decorator(f):
        @wraps(f)
        def wrapped_f(self, *args, **kwargs):
            argments = list(
                (k, v.default)
                for k, v in inspect.signature(f).parameters.items()
            )
            assert argments[0][0] == "self"
            argments = argments[1:]

            params = {}
            for (k, _), v in zip(argments[: len(args)], args):
                params[k] = v
            for k, v in argments[len(args):]:
                params[k] = kwargs.get(k, v)

            for ei in exclude:
                params.pop(ei)

            if not hasattr(self, "_arguments"):
                self._arguments = {}
            self._arguments[f.__name__] = params
            return f(self, *args, **kwargs)

        return wrapped_f

    return decorator


def save_args_cls(exclude: Sequence[str] = ()):
    def decorator(f):
        @wraps(f)
        def wrapped_f(cls, *args, **kwargs):
            argments = list(
                (k, v.default)
                for k, v in inspect.signature(f).parameters.items()
            )
            assert argments[0][0] == "cls"
            argments = argments[1:]

            params = {}
            for (k, _), v in zip(argments[: len(args)], args):
                params[k] = v
            for k, v in argments[len(args):]:
                params[k] = kwargs.get(k, v)

            for ei in exclude:
                params.pop(ei)

            fres = f(cls, *args, **kwargs)

            # 将参数结果储存在对象中
            fres._arguments = {}
            fres._arguments[f.__name__] = params
            return fres

        return wrapped_f

    return decorator

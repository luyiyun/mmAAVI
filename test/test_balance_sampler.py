from collections import Counter

import numpy as np
from mmAAVI.dataset.torch_dataset import BalanceSizeSampler


class TestBalanceSampler:

    def test_length(self):
        batch_index = np.array(["a"] * 10 + ["b"] * 100 + ["c"] * 5)
        resample_size = {"a": 10., "c": 20.}
        balance_sampler = BalanceSizeSampler(resample_size, batch_index)
        assert len(balance_sampler) == 300

    def test_loop(self):
        batch_index = np.array(["a"] * 10 + ["b"] * 100 + ["c"] * 5)
        resample_size = {"a": 10., "c": 20.}
        balance_sampler = BalanceSizeSampler(resample_size, batch_index)
        index = list(balance_sampler)
        assert len(index) == 300

        cnt = Counter([batch_index[i] for i in index])
        assert cnt["a"] == 100
        assert cnt["b"] == 100
        assert cnt["c"] == 100

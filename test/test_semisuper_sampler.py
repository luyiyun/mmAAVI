from itertools import chain
from collections import Counter

import numpy as np
from mmAAVI.dataset.torch_dataset import SemiSupervisedSampler


class TestSemiSupervisedSampler:

    def test_length(self):
        for _ in range(10):
            size = np.random.randint(10000, 20000)
            bs = np.random.randint(2, 512)
            if bs % 2 == 1:
                bs += 1
            sslabel = np.random.choice([1, -1], size=size)
            sss = SemiSupervisedSampler(sslabel, bs)
            batches = list(iter(sss))
            assert len(batches) == len(sss)

            if len(batches) > 1:
                # 这些batch的大小都是相同的或者
                bss = [len(bi) for bi in batches[:-1]]
                assert len(set(bss)) == 1
                # 最后一个batch的大小<=上面的大小
                assert len(batches[-1]) <= bss[0]

            # 至多重复一个
            count = Counter(chain.from_iterable(batches))
            count2 = Counter(count.values())
            assert len(count2) <= 2
            for i in count2.keys():
                assert i in [1, 2]

            # 没有被包括进去的样本至多2个 ?
            exclu = [i for i in range(size) if i not in count.keys()]
            assert len(exclu) <= 2

            # 每个batch中都有至少两个label和两个unlabel（或者直接是0），
            # 就是不能是1
            for batchi in batches:
                batchi = sslabel[batchi]
                count3 = Counter(batchi)
                assert count3[1] != 1
                assert count3[-1] != 1

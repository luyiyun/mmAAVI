import torch
import torch.distributions as D

from mmAAVI.model.block import (
    VanillaMLP, DistributionMLP, DistributionDotDecoder
)
from mmAAVI.model.utils import ZILN


class TestVanillaMLP:

    def test_dim_right(self):
        n, inc, outc = 2, 10, 2
        hiddens = [5, 5]
        net = VanillaMLP(inc, outc, hiddens)
        inpt = torch.rand(n, inc)
        out = net(inpt)
        assert out.shape == (n, outc)

    def test_inpt_with_covs(self):
        n, inc, outc = 2, 10, 2
        hiddens = [5, 5]
        inpt = torch.rand(n, inc)

        ccov_dims, dcov_dims = [3, 1], [2, 4]
        ccovs = [torch.randn(n, i) for i in ccov_dims]
        dcovs = [torch.randint(i, size=(n,), dtype=torch.long) for i in dcov_dims]

        net = VanillaMLP(inc, outc, hiddens, continue_cov_dims=ccov_dims)
        out = net(inpt, continue_covs=ccovs)
        assert out.shape == (n, outc)
        net = VanillaMLP(inc, outc, hiddens, discrete_cov_dims=dcov_dims)
        out = net(inpt, discrete_covs=dcovs)
        assert out.shape == (n, outc)
        net = VanillaMLP(inc, outc, hiddens,
                         continue_cov_dims=ccov_dims,
                         discrete_cov_dims=dcov_dims)
        out = net(inpt, continue_covs=ccovs, discrete_covs=dcovs)
        assert out.shape == (n, outc)


class TestDistributionMLP:

    def test_out_true_dist(self):
        n, inc, outc = 2, 10, 2
        hiddens = [5, 5]
        inpt = torch.rand(n, inc)

        ccov_dims, dcov_dims = [3, 1], [2, 4]
        ccovs = [torch.randn(n, i) for i in ccov_dims]
        dcovs = [torch.randint(i, size=(n,), dtype=torch.long) for i in dcov_dims]

        for dname in ["normal", "nb", "categorical", "categorical_gumbel"]:
            dist_class = {
                "normal": D.Normal,
                "nb": D.NegativeBinomial,
                "categorical": D.Categorical,
                "categorical_gumbel": D.RelaxedOneHotCategorical,
            }
            net = DistributionMLP(inc, outc, hiddens, ccov_dims, dcov_dims,
                                  distribution=dname,
                                  distribution_style="sample")
            mu, dist = net(inpt, ccovs, dcovs, temperature=1.0)

            assert mu.shape == (n, outc)
            assert isinstance(dist, dist_class[dname])
            if dname.startswith("categorical"):
                assert dist.probs.shape == (n, outc)
            else:
                assert dist.mean.shape == (n, outc)

    def test_batch_dist(self):
        n, inc, outc = 2, 10, 2
        hiddens = [5, 5]
        inpt = torch.rand(n, inc)

        ccov_dims, dcov_dims = [3, 1], [2, 4]
        ccovs = [torch.randn(n, i) for i in ccov_dims]
        dcovs = [torch.randint(i, size=(n,), dtype=torch.long) for i in dcov_dims]

        for dname in ["normal", "nb", "ziln"]:
            dist_class = {
                "normal": D.Normal,
                "nb": D.NegativeBinomial,
                "ziln": ZILN
            }
            if dname == "nb":
                net = DistributionMLP(
                    inc, outc, hiddens, [1]+ccov_dims, dcov_dims,
                    distribution=dname, distribution_style="batch"
                )
                mu, dist = net(inpt, [torch.rand(n, 1)]+ccovs, dcovs)
            else:
                net = DistributionMLP(
                    inc, outc, hiddens, ccov_dims, dcov_dims,
                    distribution=dname, distribution_style="batch"
                )
                mu, dist = net(inpt, ccovs, dcovs)

            assert mu.shape == (n, outc)
            assert isinstance(dist, dist_class[dname])
            assert dist.mean.shape == (n, outc)


class TestDDistributionDotDecoder:

    def test_out_true_dist(self):
        dist_class = {
            "normal": D.Normal,
            "nb": D.NegativeBinomial,
            "ziln": ZILN,
        }

        n, zdim, outc, nbatches = 2, 3, 10, 3
        u = torch.rand(n, zdim)
        v = torch.rand(outc, zdim)
        blabel = torch.randint(nbatches, size=(n,))

        for dname in ["normal", "nb", "ziln"]:
            net = DistributionDotDecoder(outc, nbatches, dname, "batch")
            l = torch.rand(n, 1) if dname == "nb" else None
            dist = net(u, v, b=blabel, l=l)

            assert isinstance(dist, dist_class[dname])
            assert dist.mean.shape == (n, outc)

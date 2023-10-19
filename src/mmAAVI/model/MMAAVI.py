import collections as ct
import gc
from copy import deepcopy
from itertools import combinations
from typing import (Any, Dict, Literal, Mapping, Optional, OrderedDict,
                    Sequence, Tuple, Union)

import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from ..constant import EPS
from ..dataset import MosaicData
from ..dataset.torch_dataset import TorchMapDataset
from ..dataset.utils import normalize_edges
from ..typehint import BATCH, LOSS, LOSS_W  # , EMBED, X_REC
from ..utils import save_args_cls
from .base import MMOModel
from .decoders import (BaseMultiOmicsDecoder, GraphDecoder,
                       MixtureMultiModalDecoder, MLPMultiModalDecoder)
from .discriminator import Discriminator
from .encoders import AttEncoder, AttGMMEncoder, BaseEncoder, GraphEncoder
from .train.utils import tensors_to_device

T = torch.Tensor


class MMAAVI_wo_init(MMOModel):
    @classmethod
    def load(
        cls,
        fn: str,
        init_name: Optional[str] = None,
        map_location: str = "cpu",
        **kwargs: dict[str, Any],
    ) -> "MMAAVI_wo_init":
        res = torch.load(fn, map_location=torch.device(map_location))
        state_dict = res["state_dict"]
        args = res["arguments"]
        if init_name is None:
            ks = [k for k in args.keys() if k.endswith("model")]
            if len(ks) != 1:
                raise ValueError("please give a init_name")
            init_name = ks[0]
        args_init = args[init_name]
        init_func = getattr(cls, init_name)
        args_init.update(kwargs)  # 可能有新的参数
        inst = init_func(**args_init)
        inst.load_state_dict(state_dict)
        return inst

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseMultiOmicsDecoder,
        discriminator: Optional[Discriminator] = None,
        input_with_batch: bool = True,
        disc_use_mean: bool = True,
        disc_condi_train: Optional[Literal["hard", "soft"]] = None,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

        self._iwb = input_with_batch
        self._dum = disc_use_mean
        self._dct = disc_condi_train
        self._enc_is_gmm = isinstance(self.encoder, AttGMMEncoder)
        self._dec_is_mlp = isinstance(self.decoder, MLPMultiModalDecoder)
        self._dec_is_mix = isinstance(self.decoder, MixtureMultiModalDecoder)
        self._enc_ssl = self._enc_is_gmm and getattr(
            self.encoder, "ssl", False
        )
        self._n_omics = len(self.decoder.outcs)

        self.default_weights = {
            "alpha": 20.0,
            "noise_scale": 0.0,
            "sup": 1.0,
            "rec": 1.0,
            "disc": 1.0,
            "label_smooth": 0.1,
        }
        if self._enc_is_gmm:
            self.default_weights["temperature"] = 1.0
            for s in ["z", "c", "u"]:
                self.default_weights["kl_%s" % s] = 1  # self._n_omics
        else:
            self.default_weights["kl"] = 1  # self._n_omics

    def forward(
        self,
        batch: BATCH,
        return_rec_x: bool = False,
        # rec_blabel: Optional[T] = None,
        **kwargs: Any,  # 输入一些额外需要的参数，比如temperature
    ) -> Dict[str, T]:
        inpts, masks, blabel = batch["input"], batch["mask"], batch["blabel"]

        fres = self.encoder(
            inpts,
            masks,
            discrete_covs=[blabel] if self._iwb else [],
            **kwargs,
        )

        if not return_rec_x:
            return fres

        raise NotImplementedError

    def step_enc(self, batch: BATCH) -> Tuple[Dict[str, Any], LOSS_W]:
        kwargs = {}
        if self._enc_is_gmm:
            kwargs["sslabel"] = batch.get("sslabel", None)
            kwargs["temperature"] = self.weight_values_e["temperature"]
        enc_res, lossi = self.encoder.step(
            inpts=batch["input"],
            masks=batch["mask"],
            discrete_covs=[batch["blabel"]] if self._iwb else [],
            **kwargs,
        )
        if "zsample" not in enc_res:
            enc_res["zsample"] = enc_res["z"].rsample()

        loss_res = {}
        for k, v in lossi.items():
            if k.startswith("kl_z"):
                loss_res[k] = (v, self.weight_values_e["kl_z"])
            elif k.startswith("kl_c"):
                loss_res[k] = (v, self.weight_values_e["kl_c"])
            elif k.startswith("kl_u"):
                loss_res[k] = (v, self.weight_values_e["kl_u"])
            elif k == "kl":
                loss_res[k] = (v, self.weight_values_e["kl"])
            elif k == "sup":
                loss_res[k] = (v, self.weight_values_e["sup"])
        return enc_res, loss_res

    def step_dec(
        self, batch: BATCH, enc_res: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], LOSS_W]:
        # 这样避免总是运行rsample
        zsample = enc_res["zsample"]
        if self._dec_is_mlp:
            rec_h, rec_d, lossi = self.decoder.step(
                inpt=zsample,
                oupt=batch["output"],
                masks=batch["mask"],
                discrete_covs=[batch["blabel"]],
            )
            return {"hidden": rec_h, "dist": rec_d}, lossi

        oupts = batch["output"]
        vsample = enc_res["vsample"]
        # 将v进行拆分
        vsamples = ct.OrderedDict()
        start = 0
        for k, outi in oupts.items():
            end = start + outi.shape[1]
            vsamples[k] = vsample[start:end]
            start = end

        if self._dec_is_mix:
            # 使用MixutreMultiModalDecoder
            rec_d, lossi = self.decoder.step(
                u=zsample,
                vs=vsamples,
                oupt=oupts,
                masks=batch["mask"],
                discrete_covs=[batch["blabel"]],
                weight_dot=self.weight_values_e["weight_dot"],
                weight_mlp=self.weight_values_e["weight_mlp"],
            )
        else:
            library_size = ct.OrderedDict(
                (k, outi.sum(dim=1, keepdim=True)) for k, outi in oupts.items()
            )
            # 使用DotMultiModalDecoder
            rec_d, lossi = self.decoder.step(
                u=zsample,
                vs=vsamples,
                oupt=oupts,
                masks=batch["mask"],
                b=batch["blabel"],
                ls=library_size,
            )
        lossi = {k: (v, self.weight_values_e["rec"]) for k, v in lossi.items()}
        return {"dist": rec_d}, lossi

    def step_disc(
        self, batch: BATCH, enc_res: Dict[str, Any], dec_res: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], LOSS_W]:
        if self.discriminator is not None:
            alphai = self.weight_values_e["alpha"]
            noise_scale_i = self.weight_values_e["noise_scale"]
            disc_inpt = enc_res["z"].mean if self._dum else enc_res["zsample"]
            ccovs, dcovs = [], []
            if noise_scale_i != 0.0:
                stdi = disc_inpt.detach().std(dim=0, keepdim=True)
                disc_inpt = (
                    disc_inpt
                    + torch.randn_like(disc_inpt) * noise_scale_i * stdi
                )
            if self._dct is not None:
                cprobs = enc_res["c"].probs.detach()
                if self._dct == "hard":
                    dcovs.append(cprobs.argmax(dim=1))
                elif self._dct == "soft":
                    ccovs.append(cprobs)
            weights = (
                batch["disc_weights"] if "disc_weights" in batch else None
            )
            logits, lossi = self.discriminator.step(
                disc_inpt,
                label=batch["dlabel"],
                continue_covs=ccovs,
                discrete_covs=dcovs,
                weights=weights,
                alpha=alphai,
                label_smooth=self.weight_values_e["label_smooth"],
            )
            lossi = {
                k: (v, self.weight_values_e["disc"]) for k, v in lossi.items()
            }
            return {"logits": logits}, lossi

        return {}, {}

    def step_sum_loss(self, losses: LOSS_W) -> LOSS:
        # 计算total和metric
        loss_res, loss_metric = {}, 0.0
        for k, (v, w) in losses.items():
            if not k.startswith("disc"):
                loss_metric += v * w
            loss_res[k] = v
        loss_total = loss_metric.clone()
        for k, (v, w) in losses.items():
            if k.startswith("disc"):
                loss_total += v * w
        loss_res["total"] = loss_total
        loss_res["metric"] = loss_metric

        return loss_res

    def step(
        self, batch: BATCH
    ) -> Tuple[
        Dict[str, Any],  # enc_res
        Dict[str, Any],  # dec_res
        Dict[str, Any],  # disc_res
        LOSS,  # losses
    ]:
        losses = {}  # dict[str, [T, float]]
        # -------------------------------------------------------------------
        # encoder，计算KL项的loss
        # -------------------------------------------------------------------
        enc_res, loss_enc = self.step_enc(batch)
        losses.update(loss_enc)

        # -------------------------------------------------------------------
        # 采样，decoder
        # -------------------------------------------------------------------
        dec_res, loss_dec = self.step_dec(batch, enc_res)
        losses.update(loss_dec)

        # -------------------------------------------------------------------
        # discriminate
        # -------------------------------------------------------------------
        disc_res, loss_disc = self.step_disc(batch, enc_res, dec_res)
        losses.update(loss_disc)

        # -------------------------------------------------------------------
        # loss summarization
        # -------------------------------------------------------------------
        losses = self.step_sum_loss(losses)  # dict[str, T]

        return enc_res, dec_res, disc_res, losses

    def attention(
        self,
        dat: MosaicData,
        batch_size: int = 256,
        num_workers: int = 1,
        device: str = "cuda:0",
        verbose: int = 1,
    ) -> T:
        if not getattr(dat, "_prepared", False):
            raise ValueError("请先运行dat.prepare()")
        loader = dat.to_dataloader(batch_size, num_workers, False, False)

        device = torch.device(device)
        self.to(device)

        self.eval()
        with torch.no_grad():
            att = []
            for batch in tqdm(loader, desc="Attention: ", disable=verbose < 1):
                batch = tensors_to_device(batch, device)
                inps, masks, blabel = (
                    batch["input"],
                    batch["mask"],
                    batch["blabel"],
                )
                atti = self.encoder.attention(
                    inps, masks, discrete_covs=[blabel] if self._iwb else []
                )
                att.append(atti)
        att = torch.cat(att, dim=0)
        return att


class MMAAVI_w_disc(MMAAVI_wo_init):
    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseMultiOmicsDecoder,
        nbatches: int,
        nclusters: int = 0,
        dim_z: int = 30,
        hiddens_disc: Optional[Sequence[int]] = (256, 256),
        act: str = "relu",
        bn: bool = True,
        dp: float = 0.2,
        spectral_norm: bool = False,
        input_with_batch: bool = True,
        disc_use_mean: bool = True,
        disc_condi_train: Optional[Literal["hard", "soft"]] = None,
    ) -> None:
        if hiddens_disc is None:
            discriminator = None
        else:
            ccdims = [nclusters] if disc_condi_train == "soft" else []
            dcdims = [nclusters] if disc_condi_train == "hard" else []
            discriminator = Discriminator(
                inc=dim_z,
                outc=nbatches,
                hiddens=hiddens_disc,
                continue_cov_dims=ccdims,
                discrete_cov_dims=dcdims,
                act=act,
                bn=bn,
                dp=dp,
                criterion="ce",
                spectral_norm=spectral_norm,
                gradient_reversal=True,
            )

        super().__init__(
            encoder,
            decoder,
            discriminator,
            input_with_batch,
            disc_use_mean,
            disc_condi_train,
        )


class MMAAVI(MMAAVI_w_disc):

    """mixture decoder and graph vae"""

    def load_pretrain(
        self,
        fn: str,
        except_prefixs: Sequence[str] = (),
        device: str = "cuda:0",
    ):
        res = torch.load(fn, map_location=torch.device(device))
        state_dict = res["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            for prefix in except_prefixs:
                if k.startswith(prefix):
                    break
            else:
                new_state_dict[k] = v
        ms_keys, unex_keys = self.load_state_dict(new_state_dict, strict=False)
        print("missing keys: " + ",".join(ms_keys))
        print("unexpected keys: " + ",".join(unex_keys))
        return self

    @classmethod
    @save_args_cls()
    def att_enc_model(
        cls,
        nbatches: int,
        dim_inputs: OrderedDict[str, int],
        dim_outputs: OrderedDict[str, int],
        dim_z: int = 30,
        dim_enc_middle: int = 200,
        hiddens_enc_unshared: Sequence[int] = (256, 256),
        hiddens_enc_z: Sequence[int] = (256,),
        hiddens_dec: Sequence[int] = (256, 256),
        hiddens_disc: Optional[Sequence[int]] = (256, 256),
        distributions: Union[str, Mapping[str, str]] = "nb",
        distributions_style: Union[str, Mapping[str, str]] = "batch",
        bn: bool = True,
        act: str = "relu",
        dp: float = 0.2,
        disc_bn: Optional[bool] = None,
        disc_use_mean: bool = False,
        omic_embed: bool = False,
        omic_embed_train: bool = False,
        spectral_norm: bool = False,
        input_with_batch: bool = False,
        reduction: str = "sum",
        graph_encoder_init_zero: bool = True,
        graph_decoder_whole: bool = False,
    ):
        encoder = AttEncoder(
            incs=dim_inputs,
            outc=dim_z,
            continue_cov_dims=[],
            discrete_cov_dims=[nbatches] if input_with_batch else [],
            hiddens_unshare=hiddens_enc_unshared,
            nlatent_middle=dim_enc_middle,
            hiddens_shared=hiddens_enc_z,
            act=act,
            bn=bn,
            dp=dp,
            reduction=reduction,
            omic_embed=omic_embed,
            omic_embed_train=omic_embed_train,
        )
        return cls(
            encoder=encoder,
            nbatches=nbatches,
            nclusters=0,
            dim_outputs=dim_outputs,
            dim_z=dim_z,
            hiddens_dec=hiddens_dec,
            hiddens_disc=hiddens_disc,
            distributions=distributions,
            distributions_style=distributions_style,
            bn=bn,
            act=act,
            dp=dp,
            disc_bn=disc_bn,
            disc_condi_train=None,
            disc_use_mean=disc_use_mean,
            spectral_norm=spectral_norm,
            input_with_batch=input_with_batch,
            reduction=reduction,
            graph_encoder_init_zero=graph_encoder_init_zero,
            graph_decoder_whole=graph_decoder_whole,
        )

    @classmethod
    @save_args_cls()
    def att_gmm_enc_model(
        cls,
        nbatches: int,
        nclusters: int,
        dim_inputs: OrderedDict[str, int],
        dim_outputs: OrderedDict[str, int],
        dim_z: int = 30,
        dim_u: int = 30,
        dim_enc_middle: int = 200,
        hiddens_enc_unshared: Sequence[int] = (256, 256),
        hiddens_enc_z: Sequence[int] = (256,),
        hiddens_enc_c: Sequence[int] = (50,),
        hiddens_enc_u: Sequence[int] = (50,),
        hiddens_dec: Sequence[int] = (256, 256),
        hiddens_prior: Sequence[int] = (),
        hiddens_disc: Optional[Sequence[int]] = (256, 256),
        distributions: Union[str, Mapping[str, str]] = "nb",
        distributions_style: Union[str, Mapping[str, str]] = "batch",
        bn: bool = True,
        act: str = "lrelu",
        dp: float = 0.2,
        disc_bn: Optional[bool] = True,
        disc_condi_train: Optional[str] = None,
        disc_use_mean: bool = False,
        c_reparam: bool = True,
        c_prior: Optional[Sequence[float]] = None,
        omic_embed: bool = False,
        omic_embed_train: bool = False,
        spectral_norm: bool = False,
        input_with_batch: bool = False,
        reduction: str = "sum",
        semi_supervised: bool = False,
        graph_encoder_init_zero: bool = True,
        graph_decoder_whole: bool = False,
    ):
        encoder = AttGMMEncoder(
            incs=dim_inputs,
            zdim=dim_z,
            cdim=nclusters,
            udim=dim_u,
            continue_cov_dims=[],
            discrete_cov_dims=[nbatches] if input_with_batch else [],
            hiddens_unshare=hiddens_enc_unshared,
            nlatent_middle=dim_enc_middle,
            hiddens_z=hiddens_enc_z,
            hiddens_c=hiddens_enc_c,
            hiddens_u=hiddens_enc_u,
            hiddens_prior=hiddens_prior,
            act=act,
            bn=bn,
            dp=dp,
            reduction=reduction,
            omic_embed=omic_embed,
            omic_embed_train=omic_embed_train,
            c_reparam=c_reparam,
            semi_supervised=semi_supervised,
            c_prior=c_prior,
        )
        return cls(
            encoder=encoder,
            nbatches=nbatches,
            nclusters=nclusters,
            dim_outputs=dim_outputs,
            dim_z=dim_z,
            hiddens_dec=hiddens_dec,
            hiddens_disc=hiddens_disc,
            distributions=distributions,
            distributions_style=distributions_style,
            bn=bn,
            act=act,
            dp=dp,
            disc_bn=disc_bn,
            disc_condi_train=disc_condi_train,
            disc_use_mean=disc_use_mean,
            spectral_norm=spectral_norm,
            input_with_batch=input_with_batch,
            reduction=reduction,
            graph_encoder_init_zero=graph_encoder_init_zero,
            graph_decoder_whole=graph_decoder_whole,
        )

    @classmethod
    def configure_data_to_loader(
        cls,
        data: MosaicData,
        phase: str = "train",
        shuffle: bool = True,
        batch_size: Optional[int] = None,
        ss_batch_size: int = 32,
        num_workers: Optional[int] = None,
        drop_last: bool = False,
        input_use: Optional[str] = None,
        output_use: Optional[str] = None,
        obs_blabel: Optional[str] = None,
        obs_dlabel: Optional[str] = None,
        obs_sslabel: Optional[str] = None,
        obs_sslabel_full: Optional[str] = None,
        impute_miss: bool = False,
        sslabel_codes: Optional[Sequence[Any]] = None,
        resample: Optional[
            Union[Literal["min", "max"], dict[str, float]]
        ] = "max",
        net_use: Optional[str] = None,
        net_batch_size: Union[int, float] = 10000,
        drop_self_loop: bool = False,
        num_negative_samples: int = 10,
    ) -> Tuple[
        DataLoader,  # dataloader
        OrderedDict[str, int],  # input_dims
        OrderedDict[str, int],  # output_dims
    ]:
        if batch_size is None:
            batch_size = int(max(data.nobs // 65, 32))  # 自动设置
        return super().configure_data_to_loader(
            data,
            phase,
            shuffle,
            batch_size,
            ss_batch_size,
            num_workers,
            drop_last,
            input_use,
            output_use,
            obs_blabel,
            obs_dlabel,
            obs_sslabel,
            obs_sslabel_full,
            impute_miss,
            sslabel_codes,
            resample,
            net_use,
            "sarr",
            net_batch_size,
            drop_self_loop,
            num_negative_samples=num_negative_samples,
        )

    def __init__(
        self,
        encoder: BaseEncoder,
        nbatches: int,
        nclusters: int,
        dim_outputs: OrderedDict[str, int],
        dim_z: int = 30,
        hiddens_dec: Sequence[int] = (256, 256),
        hiddens_disc: Optional[Sequence[int]] = (256, 256),
        distributions: Union[str, Mapping[str, str]] = "nb",
        distributions_style: Union[str, Mapping[str, str]] = "batch",
        bn: bool = True,
        act: str = "relu",
        dp: float = 0.2,
        disc_bn: Optional[bool] = None,
        disc_condi_train: Optional[str] = "hard",
        disc_use_mean: bool = False,
        spectral_norm: bool = False,
        input_with_batch: bool = True,
        reduction: str = "sum",
        graph_encoder_init_zero: bool = True,
        graph_decoder_whole: bool = False,
    ) -> None:
        decoder = MixtureMultiModalDecoder(
            inc=dim_z,
            outcs=dim_outputs,
            hiddens=hiddens_dec,
            continue_cov_dims=[],
            discrete_cov_dims=[nbatches],
            act=act,
            bn=bn,
            dp=dp,
            reduction=reduction,
            distributions=distributions,
            distributions_style=distributions_style,
        )

        if disc_bn is None:
            disc_bn = bn
        super().__init__(
            encoder,
            decoder,
            nbatches,
            nclusters,
            dim_z,
            hiddens_disc,
            act,
            disc_bn,
            dp,
            spectral_norm,
            input_with_batch,
            disc_use_mean,
            disc_condi_train,
        )

        self._flag_gdw = graph_decoder_whole
        vnum = sum(dim_outputs.values())
        self.graph_enc = GraphEncoder(
            vnum, dim_z, reduction, zero_init=graph_encoder_init_zero
        )
        self.graph_dec = GraphDecoder()
        self.default_weights["kl_graph"] = 0.01
        self.default_weights["rec_graph"] = len(dim_outputs)
        self.default_weights["weight_dot"] = 0.9
        self.default_weights["weight_mlp"] = 0.1

    def step_enc(self, batch: BATCH) -> Tuple[Dict[str, Any], LOSS_W]:
        enc_res, lossi = super().step_enc(batch)

        sidx, tidx, esign, _, enorm = batch["varp"]
        v, loss_g = self.graph_enc.step(sidx, tidx, enorm, esign)
        v = v["z"]
        enc_res["v"] = v
        enc_res["vsample"] = v.rsample()
        lossi["kl_graph"] = (
            loss_g["kl_graph"],
            self.weight_values_e["kl_graph"],
        )
        return enc_res, lossi

    def step_dec(
        self, batch: BATCH, enc_res: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], LOSS]:
        dec_res, lossi = super().step_dec(batch, enc_res)
        # NOTE: glue其实每次只计算了一部分的edges
        vsample = enc_res["vsample"]
        if self._flag_gdw:
            sidx, tidx, esign, ewt, _ = batch["varp"]
            _, loss_g = self.graph_dec.step(vsample, sidx, tidx, esign, ewt)
        else:
            sidx_i, tidx_i, es_i, ewt_i = batch["sample_graph"][:4]
            _, loss_g = self.graph_dec.step(
                vsample, sidx_i, tidx_i, es_i, ewt_i
            )
        lossi["rec_graph"] = (
            loss_g["rec_graph"],
            self.weight_values_e["rec_graph"],
        )
        return dec_res, lossi

    def differential(
        self,
        dat: MosaicData,
        obs_label: str,
        used_net: str,
        obs_blabel: str = "_batch",
        nsamples: int = 5000,
        batch_size: int = 256,
        num_workers: int = 0,
        device: str = "cuda:0",
        verbose: int = 1,
        input_use: Optional[str] = None,
        output_use: Optional[str] = None,
        # save_fn: Optional[str] = None,
        # summary: bool = True,
        rec_type: str = "mean",
        weight_dot: float = 0.9,
        weight_mlp: float = 0.1,
        method: str = "macro",
    ):
        """
        if save_dir is None, the all results will be return; otherwise, save
        the results into multiple files in the save_dir. if the data is big,
        save_dir is helpful.
        """
        assert rec_type in ["mean", "sample", "h"]
        assert method in ["macro", "paired"]

        device = torch.device(device)
        self.to(device)
        self.eval()

        # Preprocess the Graph, and the embeds will be used directly.
        # firstly, get the coo format of graph with
        net = dat.nets[used_net]
        net.as_sparse_type("csr", only_sparse=False)
        nets = net.to_array(True)
        nets = nets.tocoo()
        # make net symmetric and add self-loop
        i, j, v = nets.row, nets.col, nets.data
        new_i = np.concatenate([i, j, np.arange(nets.shape[0])])
        new_j = np.concatenate([j, i, np.arange(nets.shape[1])])
        new_w = np.concatenate([v, v, np.ones(nets.shape[0])])
        new_s = 2 * (new_w > 0.0).astype(float) - 1  # NOTE: 1和-1
        new_w = np.abs(new_w)
        # get normalized edge weights
        enorm = normalize_edges((nets.shape, new_i, new_j, new_s, new_w))
        # wrap the ndarray as tensor
        new_i = torch.tensor(new_i, dtype=torch.long, device=device)
        new_j = torch.tensor(new_j, dtype=torch.long, device=device)
        new_s = torch.tensor(new_s, dtype=torch.float32, device=device)
        # new_w = torch.tensor(new_w, dtype=torch.float32)
        enorm = torch.tensor(enorm, dtype=torch.float32, device=device)
        # get the V
        V = self.graph_enc(new_i, new_j, enorm, new_s)["z"]

        results = {
            "V": (
                V.mean.detach().cpu().numpy(),
                V.stddev.detach().cpu().numpy(),
            )
        }

        # split the MosaicData by obs_label
        label = dat.obs[obs_label].values
        batch = dat.obs[obs_blabel].values
        batch_uni = np.unique(batch)

        decoder = self.decoder

        def _split_vsample(vsample):
            vsamples = ct.OrderedDict()
            start = 0
            for k, outi in decoder.outcs.items():
                end = start + outi
                vsamples[k] = vsample[start:end]
                start = end
            return vsamples

        def _refine_label(lname):
            lname = str(lname)
            if "/" in lname:
                lname = lname.replace("/", "_")
            return lname

        def _iter_pos_neg_dat(
            dat: MosaicData,
            labels: np.ndarray,
            verbose: int = 0,
            method: str = "macro",
        ):
            label_uni = np.unique(labels)
            if method == "macro":
                for labeli in tqdm(
                    label_uni, desc="Labels: ", disable=(verbose == 0)
                ):
                    labeli_name = _refine_label(labeli)

                    # get the positive cells and negative cells
                    mask_pos = label == labeli
                    mask_neg = np.logical_not(mask_pos)
                    dat_pos = dat.select_rows(mask_pos)
                    dat_neg = dat.select_rows(mask_neg)

                    yield dat_pos, dat_neg, (labeli_name,)

            elif method == "paired":
                len_label = len(label_uni)
                for l1, l2 in tqdm(
                    combinations(label_uni, 2),
                    desc="Labels: ",
                    disable=(verbose == 0),
                    total=int(len_label * (len_label - 1) / 2),
                ):
                    l1_name = _refine_label(l1)
                    l2_name = _refine_label(l2)

                    mask_pos = label == l1
                    mask_neg = label == l2
                    dat_pos = dat.select_rows(mask_pos)
                    dat_neg = dat.select_rows(mask_neg)

                    yield dat_pos, dat_neg, (l1_name, l2_name)

            else:
                raise NotImplementedError

        # collect the results
        res_de = {}
        for dat_pos, dat_neg, lnames in _iter_pos_neg_dat(
            dat, label, verbose, method
        ):
            # wrap the MosaicData as DataLoader
            # the DataLoader has two parts: Tensor part and Graph part
            # Graph part has been preprocessed and cached
            # There is tensor part here.
            tdata_pos = TorchMapDataset(
                dat_pos,
                input_use=input_use,
                output_use=output_use,
                obs_blabel=obs_blabel,
                obs_dlabel=None,
                obs_sslabel=None,
                obs_sslabel_full=None,
                impute_miss=False,
                sslabel_codes=None,
            )
            tdata_neg = TorchMapDataset(
                dat_neg,
                input_use=input_use,
                output_use=output_use,
                obs_blabel=obs_blabel,
                obs_dlabel=None,
                obs_sslabel=None,
                obs_sslabel_full=None,
                impute_miss=False,
                sslabel_codes=None,
            )
            sampler1 = RandomSampler(
                tdata_pos, replacement=True, num_samples=nsamples
            )
            sampler2 = RandomSampler(
                tdata_neg, replacement=True, num_samples=nsamples
            )
            loader_pos = DataLoader(
                tdata_pos,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler1,
                collate_fn=tdata_pos.get_collated_fn(),
                drop_last=False,
            )
            loader_neg = DataLoader(
                tdata_neg,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler2,
                collate_fn=tdata_neg.get_collated_fn(),
                drop_last=False,
            )
            pos_blabel_categories = tdata_pos._blabel.cat.categories
            # neg_blabel_categories = tdata_neg._blabel.cat.categories

            # if use c_reparam，need to set the wight_values_e
            self.weight_values_e = deepcopy(self.default_weights)
            # use loop to get the outputs of decoders
            diff_tensors = ct.defaultdict(list)
            with torch.no_grad():
                for batch_pos, batch_neg in tqdm(
                    zip(loader_pos, loader_neg),
                    total=len(loader_pos),
                    desc="Batch: ",
                    disable=(verbose < 1),
                    leave=False,
                ):
                    batch_pos = tensors_to_device(batch_pos, device)
                    batch_neg = tensors_to_device(batch_neg, device)

                    # encode is independent from batch indices
                    enc_res_pos, _ = super().step_enc(batch_pos)
                    enc_res_neg, _ = super().step_enc(batch_neg)
                    library_size = ct.OrderedDict(
                        (
                            k,
                            torch.full(
                                (enc_res_pos["zsample"].shape[0], 1),
                                fill_value=nd,
                                dtype=torch.float32,
                                device=device,
                            ),
                        )
                        for k, nd in dat.omics_dims_dict.items()
                    )

                    for blabeli in batch_uni:
                        blabeli_t = torch.full_like(
                            batch_pos["blabel"],
                            pos_blabel_categories.get_loc(blabeli),
                        )
                        # add v into the enc_res
                        vsample = _split_vsample(V.sample())

                        hs_pos, dec_res_pos = decoder(
                            u=enc_res_pos["zsample"],
                            vs=vsample,
                            discrete_covs=[blabeli_t],
                            library_size=library_size,
                            weight_dot=weight_dot,
                            weight_mlp=weight_mlp,
                        )
                        hs_neg, dec_res_neg = decoder(
                            u=enc_res_neg["zsample"],
                            vs=vsample,
                            discrete_covs=[blabeli_t],
                            library_size=library_size,
                            weight_dot=weight_dot,
                            weight_mlp=weight_mlp,
                        )

                        for k in dec_res_pos.keys():
                            if rec_type == "mean":
                                rec_pos_k = dec_res_pos[k].mean
                                rec_neg_k = dec_res_neg[k].mean
                            elif rec_type == "sample":
                                rec_pos_k = dec_res_pos[k].sample()
                                rec_neg_k = dec_res_neg[k].sample()
                            elif rec_type == "h":
                                rec_pos_k = hs_pos[k]
                                rec_neg_k = hs_neg[k]
                            else:
                                raise NotImplementedError
                            diff_tensors[k].append(rec_pos_k - rec_neg_k)

            res_de_i = {}
            for k, tensors in diff_tensors.items():
                tensors = torch.cat(tensors, dim=0)
                pvalue = (tensors > 0).mean(dim=0)
                md = torch.mean(tensors, dim=0)
                dfi = pd.DataFrame(
                    {
                        "md": md.detach().cpu().numpy(),
                        "p": pvalue.detach().cpu().numpy(),
                        "bf": torch.logit(pvalue, EPS).detach().cpu().numpy(),
                    },
                    index=dat.var.index[dat.var["_omics"] == k].values,
                )
                res_de_i[k] = dfi
            res_de[lnames] = res_de_i

            # avoid the memory explosion but slow
            del (
                diff_tensors,
                loader_pos,
                loader_neg,
                tdata_pos,
                tdata_neg,
                dat_pos,
                dat_neg,
            )
            gc.collect()

        # summarize the results for every label
        if method == "macro":
            res_de = {k[0]: v for k, v in res_de.items()}
        elif method == "paired":
            raise NotImplementedError
            # # get the all labels
            # label_names = set()
            # for ks in res_de.keys():
            #     label_names.update(ks)
            # # iter each label
            # for labeli in label_names:
            #     df = None
            #     for (k1, k2), dfi in res_de.items():
            #         if labeli == k1:
            #             pass
            #         elif labeli == k2:
            #             dfi["p"] = 1 - df["p"]
            #             dfi["md"] = -dfi["md"]
            #             dfi["bf"] = -dfi["bf"]
            #         else:
            #             continue
            #         if df is None:
            #             df = dfi
            #         else:
            #             df["p"] =
            # pass
        else:
            raise NotImplementedError

        results["de"] = res_de
        return results

    def differential_semi(
        self,
        dat: MosaicData,
        label_codes: Sequence[str],
        # obs_label: str,
        used_net: str,
        # obs_blabel: str = "_batch",
        nsamples: int = 5000,
        batch_size: int = 256,
        # num_workers: int = 0,
        device: str = "cuda:0",
        verbose: int = 2,
        # input_use: Optional[str] = None,
        # output_use: Optional[str] = None,
        # save_dir: Optional[str] = None,
        # summary: bool = True,
        # use_mean: bool = True,
        weight_dot=1.0,
        weight_mlp=0.0,
    ):
        """
        if save_dir is None, the all results will be return; otherwise, save
        the results into multiple files in the save_dir. if the data is big,
        save_dir is helpful.
        """
        device = torch.device(device)
        args_df = {"dtype": torch.float32, "device": device}
        args_di = {"dtype": torch.long, "device": device}
        self.to(device)
        self.eval()

        # Preprocess the Graph, and the embeds will be used directly.
        # firstly, get the coo format of graph with
        net = dat.nets[used_net]
        net.as_sparse_type("csr", only_sparse=False)
        nets = net.to_array(True)
        nets = nets.tocoo()
        # make net symmetric and add self-loop
        i, j, v = nets.row, nets.col, nets.data
        new_i = np.concatenate([i, j, np.arange(nets.shape[0])])
        new_j = np.concatenate([j, i, np.arange(nets.shape[1])])
        new_w = np.concatenate([v, v, np.ones(nets.shape[0])])
        new_s = 2 * (new_w > 0.0).astype(float) - 1  # NOTE: 1和-1
        new_w = np.abs(new_w)
        # get normalized edge weights
        enorm = normalize_edges((nets.shape, new_i, new_j, new_s, new_w))
        # wrap the ndarray as tensor
        new_i = torch.tensor(new_i, dtype=torch.long, device=device)
        new_j = torch.tensor(new_j, dtype=torch.long, device=device)
        new_s = torch.tensor(new_s, dtype=torch.float32, device=device)
        # new_w = torch.tensor(new_w, dtype=torch.float32)
        enorm = torch.tensor(enorm, dtype=torch.float32, device=device)
        # get the V
        V = self.graph_enc(new_i, new_j, enorm, new_s)["z"]

        encoder = self.encoder
        decoder = self.decoder
        u = Normal(
            torch.zeros(encoder.udim, **args_df),
            torch.ones(encoder.udim, **args_df),
        )
        nc = encoder.cdim

        def _split_vsample(vsample):
            vsamples = ct.OrderedDict()
            start = 0
            for k, outi in decoder.outcs.items():
                end = start + outi
                vsamples[k] = vsample[start:end]
                start = end
            return vsamples

        def _one_hot(ind):
            return torch.eye(nc, **args_df)[ind, :]

        def _sample_int_without(i, n):
            inds = list(range(nc))
            inds.remove(i)
            return torch.tensor(np.random.choice(inds, n), **args_di)

        def _level_batch(csamplei, usamplei, batch_t, library_size):
            _, z = encoder.p_z_cu(torch.cat([csamplei, usamplei], dim=1))
            zsample = z.sample()
            vsample = _split_vsample(V.sample())
            _, res_d = decoder(
                u=zsample,
                vs=vsample,
                discrete_covs=[batch_t],
                library_size=library_size,
                weight_dot=weight_dot,
                weight_mlp=weight_mlp,
            )
            return {k: v.mean for k, v in res_d.items()}

        def _level_batch_labeli(labeli: int, usamplei, batch_t, library_size):
            pos_c = torch.full((usamplei.shape[0],), labeli, **args_di)
            neg_c = _sample_int_without(labeli, usamplei.shape[0])
            pos_c, neg_c = _one_hot(pos_c), _one_hot(neg_c)

            pos_res = _level_batch(pos_c, usamplei, batch_t, library_size)
            neg_res = _level_batch(neg_c, usamplei, batch_t, library_size)

            res = {k: pos_res[k] - neg_res[k] for k in pos_res.keys()}
            return res

        def _level_batch_batches(labeli: int, usamplei, library_size):
            res = ct.defaultdict(list)
            for batchi in range(dat.nbatch):
                batch_t = torch.full(
                    (usamplei.shape[0],),
                    batchi,
                    dtype=torch.long,
                    device=device,
                )
                resi = _level_batch_labeli(
                    labeli, usamplei, batch_t, library_size
                )
                for k, v in resi.items():
                    res[k].append(v)
            return {k: torch.cat(v) for k, v in res.items()}

        results = {}
        for labeli, label_name in tqdm(
            enumerate(label_codes),
            total=len(label_codes),
            desc="Label: ",
            disable=(verbose == 0),
        ):
            res = ct.defaultdict(list)
            for starti in tqdm(
                range(0, nsamples, batch_size),
                desc="Batch: ",
                leave=False,
                disable=(verbose <= 1),
            ):
                endi = min(nsamples, starti + batch_size)
                nsamplei = endi - starti

                usample = u.sample((nsamplei,))
                library_size = ct.OrderedDict(
                    (
                        k,
                        torch.full(
                            (nsamplei, 1),
                            fill_value=nd,
                            dtype=torch.float32,
                            device=device,
                        ),
                    )
                    for k, nd in dat.omics_dims_dict.items()
                )
                resi = _level_batch_batches(labeli, usample, library_size)
                for k, v in resi.items():
                    res[k].append(v.detach().cpu().numpy())

            diff_dfs = {}
            for k in res.keys():
                arr = np.concatenate(res[k])
                pvalue = (arr > 0).mean(axis=0)
                dfk = pd.DataFrame(
                    {
                        "md": np.median(arr, axis=0),
                        "p": pvalue,
                        "bf": np.log(pvalue + EPS) - np.log(1 - pvalue + EPS),
                    },
                    index=dat.var.index[dat.var["_omics"] == k].values,
                )
                diff_dfs[k] = dfk

            results[label_name] = diff_dfs

        return results, (
            V.mean.detach().cpu().numpy(),
            V.stddev.detach().cpu().numpy(),
        )

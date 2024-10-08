import warnings
from typing import Sequence, Optional, Union, Mapping, Tuple, Literal, Any
from itertools import combinations

import numpy as np
import pandas as pd
import torch

# from torch.utils.data import DataLoader
from mudata import MuData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .preprocess import merge_obs_from_all_modalities
from .dataset import get_dataloader, GraphDataLoader
from .network import MMAAVINET
from .trainer import Trainer
from .train_utils.utils import tensors_to_device
from .utils import setup_seed


EPS = 1e-7


def encode_obs(
    mdata: MuData,
    read_key: str,
    save_key: str,
    nan_as: Optional[Tuple[float, int]] = None,
) -> LabelEncoder:
    if save_key in mdata.obs.columns:
        warnings.warn(
            f"{save_key} exists, it will " "be cover by intermediate columns."
        )
    enc = LabelEncoder()
    array = mdata.obs[read_key].values
    if nan_as is not None:
        value_mask = ~pd.isnull(array)
        codes = np.full_like(array, -1, dtype=int)
        codes[value_mask] = enc.fit_transform(array[value_mask])
        mdata.obs[save_key] = codes
    else:
        mdata.obs[save_key] = enc.fit_transform(array)
    return enc


class MMAAVI:
    def __init__(
        self,
        mixture_embeddings: bool = True,
        decoder_style: Literal["mlp", "glue", "mixture"] = "mixture",
        disc_alignment: bool = True,
        dim_z: int = 30,
        dim_u: int = 30,
        dim_c: Optional[int] = None,
        dim_enc_middle: int = 200,
        hiddens_enc_unshared: Sequence[int] = (256, 256),
        hiddens_enc_z: Sequence[int] = (256,),
        hiddens_enc_c: Sequence[int] = (50,),
        hiddens_enc_u: Sequence[int] = (50,),
        hiddens_dec: Optional[Sequence[int]] = (
            256,
            256,
        ),  # may be None in glue-style decoder
        hiddens_prior: Sequence[int] = (),
        hiddens_disc: Optional[Sequence[int]] = (256, 256),
        distributions: Union[str, Mapping[str, str]] = "nb",
        distributions_style: Union[str, Mapping[str, str]] = "batch",
        bn: bool = True,
        act: str = "lrelu",
        dp: float = 0.2,
        disc_bn: Optional[bool] = True,
        disc_condi_train: Optional[str] = None,
        disc_on_mean: bool = True,
        disc_criterion: Literal["ce", "bce", "focal"] = "ce",
        c_reparam: bool = True,
        c_prior: Optional[Sequence[float]] = None,
        omic_embed: bool = False,
        omic_embed_train: bool = False,
        spectral_norm: bool = False,
        input_with_batch: bool = False,
        reduction: str = "sum",
        # semi_supervised: bool = False,
        graph_encoder_init_zero: bool = True,
        # graph_decoder_whole: bool = False,
        temperature: float = 1.0,
        disc_gradient_weight: float = 20.0,
        label_smooth: float = 0.1,
        focal_alpha: float = 2.0,
        focal_gamma: float = 1.0,
        # 0.9 for overlapped, 0.99 for diagonal integration
        mix_dec_dot_weight: Optional[float] = None,
        seed: Optional[int] = 0,
        valid_size: Tuple[int, float] = 0.1,
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        batch_key: str = "batch",
        dlabel_key: Optional[str] = None,
        sslabel_key: Optional[str] = None,
        # TODO: drop_last?
        balance_sample: Optional[
            Union[Literal["max", "min", "mean"], int]
        ] = "max",
        batch_size: Optional[int] = None,
        ss_label_ratio: float = 0.2,
        num_workers: int = 0,
        pin_memory: bool = True,
        net_key: Optional[str] = None,
        graph_batch_size: Union[float, int] = 10000,
        drop_self_loop: bool = False,
        num_negative_samples: int = 10,
        max_epochs: int = 300,
        device: str = "cuda:0",
        learning_rate: float = 0.002,
        optimizer: str = "rmsprop",
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = None,
        lr_schedual: Optional[str] = "reduce_on_plateau",
        sch_kwargs: dict[str, Any] = {"factor": 0.1, "patience": 5},
        sch_max_update: Optional[int] = 2,
        # valid_umap_interval: int = 3,
        # valid_show_umap: Optional[Union[str, Sequence[str]]] = None,
        checkpoint_best: bool = True,
        early_stop: bool = True,
        early_stop_patient: int = 10,
        tensorboard_dir: Optional[str] = None,
        verbose: int = 2,
        deterministic: bool = False,  # True for reproducibility
        loss_weight_kl_omics: float = 1.0,
        loss_weight_rec_omics: float = 1.0,
        loss_weight_kl_graph: float = 0.01,
        loss_weight_rec_graph: Tuple[float, str] = "nomics",
        loss_weight_disc: float = 1.0,
        loss_weight_sup: float = 1.0,
    ):
        # TODO: add docstring!
        if sslabel_key is None and mixture_embeddings:
            assert isinstance(dim_c, int) and dim_c > 1, (
                "dim_c must be an integer than greater than 1 "
                "when mixture_embeddings=True"
            )
        assert isinstance(
            valid_size, (int, float)
        ), "valid_size must be float or int"
        if isinstance(valid_size, float):
            assert (valid_size > 0) & (
                valid_size < 1
            ), "valid_size must be in [0, 1]"
        if decoder_style != "mlp":
            assert net_key is not None, "if not mlp, please give network."
        if sslabel_key is not None:
            assert (
                mixture_embeddings
            ), "if semi-supervised learning, mixture_embeddings must be true."

        self.seed_ = seed
        self.valid_size_ = valid_size
        self.input_key_ = input_key
        self.output_key_ = output_key
        self.batch_key_ = batch_key
        self.dlabel_key_ = dlabel_key
        self.sslabel_key_ = sslabel_key
        self.balance_sample_ = balance_sample
        self.batch_size_ = batch_size
        self.num_workers_ = num_workers
        self.pin_memory_ = pin_memory
        self.net_key_ = net_key
        self.graph_batch_size_ = graph_batch_size
        self.drop_self_loop_ = drop_self_loop
        self.num_negative_samples_ = num_negative_samples
        self.ss_label_ratio_ = ss_label_ratio

        self.mixture_embeddings_ = mixture_embeddings
        self.decoder_style_ = decoder_style
        self.disc_alignment_ = disc_alignment
        self.dim_z_ = dim_z
        self.dim_u_ = dim_u
        self.dim_c_ = dim_c
        self.dim_enc_middle_ = dim_enc_middle
        self.hiddens_enc_unshared_ = hiddens_enc_unshared
        self.hiddens_enc_z_ = hiddens_enc_z
        self.hiddens_enc_c_ = hiddens_enc_c
        self.hiddens_enc_u_ = hiddens_enc_u
        self.hiddens_dec_ = hiddens_dec
        self.hiddens_prior_ = hiddens_prior
        self.hiddens_disc_ = hiddens_disc
        self.distributions_ = distributions
        self.distributions_style_ = distributions_style
        self.bn_ = bn
        self.act_ = act
        self.dp_ = dp
        self.disc_bn_ = disc_bn
        self.disc_condi_train_ = disc_condi_train
        self.disc_on_mean_ = disc_on_mean
        self.disc_criterion_ = disc_criterion
        self.c_reparam_ = c_reparam
        self.c_prior_ = c_prior
        self.omic_embed_ = omic_embed
        self.omic_embed_train_ = omic_embed_train
        self.spectral_norm_ = spectral_norm
        self.input_with_batch_ = input_with_batch
        self.reduction_ = reduction
        self.semi_supervised_ = sslabel_key is not None
        self.graph_encoder_init_zero_ = graph_encoder_init_zero
        self.temperature_ = temperature
        self.disc_gradient_weight_ = disc_gradient_weight
        self.label_smooth_ = label_smooth
        self.focal_alpha_ = focal_alpha
        self.focal_gamma_ = focal_gamma
        self.mix_dec_dot_weight_ = mix_dec_dot_weight

        self.max_epochs_ = max_epochs
        self.device_ = device
        self.learning_rate_ = learning_rate
        self.optimizer_ = optimizer
        self.betas_ = betas
        self.weight_decay_ = weight_decay
        self.grad_clip_ = grad_clip
        self.lr_schedual_ = lr_schedual
        self.sch_kwargs_ = sch_kwargs
        self.sch_max_update_ = sch_max_update
        # self.valid_umap_interval_ = valid_umap_interval
        # self.valid_show_umap_ = valid_show_umap
        self.checkpoint_best_ = checkpoint_best
        self.early_stop_ = early_stop
        self.early_stop_patient_ = early_stop_patient
        self.tensorboard_dir_ = tensorboard_dir
        self.verbose_ = verbose
        self.deterministic_ = deterministic

        self.loss_weight_kl_omics_ = loss_weight_kl_omics
        self.loss_weight_rec_omics_ = loss_weight_rec_omics
        self.loss_weight_kl_graph_ = loss_weight_kl_graph
        self.loss_weight_rec_graph_ = loss_weight_rec_graph
        self.loss_weight_disc_ = loss_weight_disc
        self.loss_weight_sup_ = loss_weight_sup

    def fit(self, mdata: MuData, key_add: str = "mmAAVI") -> None:
        if self.mix_dec_dot_weight_ is None:
            # set the value by the format of mdata
            flag_overlap = False
            for on1, on2 in combinations(mdata.mod.keys(), 2):
                # samples with overlapped omics more than 10
                if (mdata.obsm[on1] & mdata.obsm[on2]).sum() > 10:
                    flag_overlap = True
                    break
            self.mix_dec_dot_weight_ = 0.9 if flag_overlap else 0.99

        # ======================= control random =======================
        if self.seed_ is not None:
            setup_seed(self.seed_, self.deterministic_)

        # TODO: add more tips
        # ======================= prepare dataset =======================
        # encode the batch label as integer codes
        merge_obs_from_all_modalities(mdata, self.batch_key_)
        batch_code_key = f"{self.batch_key_}__code"
        self.batch_enc_ = encode_obs(mdata, self.batch_key_, batch_code_key)

        # encode the sslabel as integer codes, nan as -1
        if self.sslabel_key_ is not None:
            merge_obs_from_all_modalities(mdata, self.sslabel_key_)
            sslabel_code_key = f"{self.sslabel_key_}__code"
            self.sslabel_enc_ = encode_obs(
                mdata, self.sslabel_key_, sslabel_code_key, nan_as=-1
            )
        else:
            sslabel_code_key = None

        # if dim_c is None and semi-supervised learning, auto determinate it
        if sslabel_code_key is not None:
            if self.dim_c_ is None:
                self.dim_c_ = len(self.sslabel_enc_.classes_)
            else:
                assert self.dim_c_ >= len(self.sslabel_enc_.classes_), (
                    "dim_c must be greater than the number of cell types "
                    "in sslabels when semi-supervised learning."
                )

        # split the dataset to train and valid
        train_index, valid_index = train_test_split(
            np.arange(mdata.n_obs),
            test_size=self.valid_size_,
            random_state=self.seed_,
        )
        # TODO: copy will change the order of samples !!!!!!
        mdata_train = mdata[train_index, :].copy()
        mdata_valid = mdata[valid_index, :].copy()

        # get the dataloaders
        self.batch_size_ = self.batch_size_ or int(
            max(mdata_train.n_obs // 65, 32)
        )
        loader_train = get_dataloader(
            mdata_train,
            input_key=self.input_key_,
            output_key=self.output_key_,
            batch_key=batch_code_key,
            dlabel_key=self.dlabel_key_,
            sslabel_key=sslabel_code_key,
            batch_size=self.batch_size_,
            shuffle=True,
            num_workers=self.num_workers_,
            pin_memory=self.pin_memory_,
            # if decoder is mlp, do not need graph
            net_key=self.net_key_ if self.decoder_style_ != "mlp" else None,
            graph_batch_size=self.graph_batch_size_,
            drop_self_loop=self.drop_self_loop_,
            num_negative_samples=self.num_negative_samples_,
            balance_sample_size=self.balance_sample_,
            label_ratio=self.ss_label_ratio_,
            repeat_sample=True,
            drop_last=True,
        )
        loader_valid = get_dataloader(
            mdata_valid,
            input_key=self.input_key_,
            output_key=self.output_key_,
            batch_key=batch_code_key,
            dlabel_key=self.dlabel_key_,
            sslabel_key=sslabel_code_key,
            batch_size=self.batch_size_,
            shuffle=False,
            num_workers=self.num_workers_,
            pin_memory=self.pin_memory_,
            net_key=self.net_key_ if self.decoder_style_ != "mlp" else None,
            graph_batch_size=self.graph_batch_size_,
            drop_self_loop=self.drop_self_loop_,
            num_negative_samples=self.num_negative_samples_,
            label_ratio=self.ss_label_ratio_,
            # Set repeat_sample=False. During validation, batches will be
            #   reorganized to balance labeled and unlabeled samples, but
            #   labeled samples will not be resampled to increase their
            #   quantity.
            repeat_sample=False,
            balance_sample_size=None,  # do not balance sample when valid
            # TODO: does valid dataset need balanced sampler
            #   and semisupervised sampler?
            drop_last=False,
        )

        # ======================= prepare model =======================
        if not hasattr(self, "model_"):  # can reuse trained model
            dim_inputs, dim_outputs = {}, {}
            for k, adatai in mdata.mod.items():
                dim_inputs[k] = (
                    adatai.obsm[self.input_key_].shape[1]
                    if self.input_key_ is not None
                    else adatai.X.shape[1]
                )
                dim_outputs[k] = (
                    adatai.obsm[self.output_key_].shape[1]
                    if self.output_key_ is not None
                    else adatai.X.shape[1]
                )
            nbatches = mdata.obs[batch_code_key].unique().shape[0]
            self.model_ = MMAAVINET(
                dim_inputs=dim_inputs,
                dim_outputs=dim_outputs,
                nbatches=nbatches,
                mixture_embeddings=self.mixture_embeddings_,
                decoder_style=self.decoder_style_,
                disc_alignment=self.disc_alignment_,
                dim_z=self.dim_z_,
                dim_u=self.dim_u_,
                dim_c=self.dim_c_,
                dim_enc_middle=self.dim_enc_middle_,
                hiddens_enc_unshared=self.hiddens_enc_unshared_,
                hiddens_enc_z=self.hiddens_enc_z_,
                hiddens_enc_c=self.hiddens_enc_c_,
                hiddens_enc_u=self.hiddens_enc_u_,
                hiddens_dec=self.hiddens_dec_,
                hiddens_prior=self.hiddens_prior_,
                hiddens_disc=self.hiddens_disc_,
                distributions=self.distributions_,
                distributions_style=self.distributions_style_,
                bn=self.bn_,
                act=self.act_,
                dp=self.dp_,
                disc_bn=self.disc_bn_,
                disc_condi_train=self.disc_condi_train_,
                disc_on_mean=self.disc_on_mean_,
                disc_criterion=self.disc_criterion_,
                c_reparam=self.c_reparam_,
                c_prior=self.c_prior_,
                omic_embed=self.omic_embed_,
                omic_embed_train=self.omic_embed_train_,
                spectral_norm=self.spectral_norm_,
                input_with_batch=self.input_with_batch_,
                reduction=self.reduction_,
                semi_supervised=self.semi_supervised_,
                graph_encoder_init_zero=self.graph_encoder_init_zero_,
                # graph_decoder_whole=self.graph_decoder_whole_,
                temperature=self.temperature_,
                disc_gradient_weight=self.disc_gradient_weight_,
                label_smooth=self.label_smooth_,
                focal_alpha=self.focal_alpha_,
                focal_gamma=self.focal_gamma_,
                mix_dec_dot_weight=self.mix_dec_dot_weight_,
                loss_weight_kl_omics=self.loss_weight_kl_omics_,
                loss_weight_rec_omics=self.loss_weight_rec_omics_,
                loss_weight_kl_graph=self.loss_weight_kl_graph_,
                loss_weight_rec_graph=self.loss_weight_rec_graph_,
                loss_weight_disc=self.loss_weight_disc_,
                loss_weight_sup=self.loss_weight_sup_,
            )

        # ======================= training model =======================
        self.trainer_ = Trainer(self.model_)
        self.train_hists_, self.train_best_ = self.trainer_.train(
            train_loader=loader_train,
            valid_loader=loader_valid,
            max_epochs=self.max_epochs_,
            device=self.device_,
            learning_rate=self.learning_rate_,
            optimizer=self.optimizer_,
            betas=self.betas_,
            weight_decay=self.weight_decay_,
            grad_clip=self.grad_clip_,
            lr_schedual=self.lr_schedual_,
            sch_kwargs=self.sch_kwargs_,
            sch_max_update=self.sch_max_update_,
            # valid_umap_interval=self.valid_umap_interval_,
            # valid_show_umap=self.valid_show_umap_,
            checkpoint_best=self.checkpoint_best_,
            early_stop=self.early_stop_,
            early_stop_patient=self.early_stop_patient_,
            tensorboard_dir=self.tensorboard_dir_,
            verbose=self.verbose_,
            # **kweights=self.kweights_,
        )

        # ======================= get the embeddings =======================
        loader_all_ = get_dataloader(
            mdata=mdata,
            input_key=self.input_key_,
            output_key=self.output_key_,
            batch_key=batch_code_key,
            dlabel_key=self.dlabel_key_,
            # do not need label when calc embeddings
            # use sslabel_key make using SemisupervisedSampler，the order will
            #   be error
            sslabel_key=None,  # sslabel_code_key,
            batch_size=self.batch_size_,
            shuffle=False,
            num_workers=self.num_workers_,
            pin_memory=self.pin_memory_,
            net_key=None,  # do not need graph when calc embeddings
            graph_batch_size=self.graph_batch_size_,
            drop_self_loop=self.drop_self_loop_,
            num_negative_samples=self.num_negative_samples_,
        )
        enc_res = self.trainer_.encode(
            loader_all_, device=self.device_, verbose=self.verbose_
        )
        for k, t in enc_res.items():
            mdata.obsm[f"{key_add}_{k}"] = t.detach().cpu().numpy()

        # if semi-spuervised learning, get the label predict in obs
        if self.sslabel_key_ is not None:
            proba = mdata.obsm[f"{key_add}_c"]
            pred_int = proba.argmax(axis=1)
            pred = self.sslabel_enc_.inverse_transform(pred_int)
            mdata.obs[f"{key_add}_ss_predict"] = pred

        # if mlp decoder，model doesn't have graph encoder and decoder
        if self.decoder_style_ == "mlp":
            return

        # ======================= calc the v embeddings =======================
        graph_data_loader = GraphDataLoader(
            mdata,
            self.net_key_,
            self.graph_batch_size_,
            self.drop_self_loop_,
            self.num_negative_samples_,
            "test",
        )
        normed_graph = tensors_to_device(
            graph_data_loader._normed_subgraph, torch.device(self.device_)
        )
        self.model_.eval()
        with torch.no_grad():
            fres = self.model_.gencoder({"normed_subgraph": normed_graph})
        vdist = fres["v"]
        vmean, vstd = vdist.mean, vdist.stddev
        mdata.varm[f"{key_add}_v"] = vmean.detach().cpu().numpy()
        mdata.varm[f"{key_add}_v_std"] = vstd.detach().cpu().numpy()

    def differential(
        self,
        mdata: MuData,
        label_key: str,
        nsamples: int = 5000,
        rec_type: str = "mean",
        method: Literal["macro", "paired"] = "macro",
        v_mean_key: str = "mmAAVI_v",
        v_std_key: str = "mmAAVI_v_std",
        key_add: str = "mmAAVI",
    ) -> None:
        assert method in ["macro", "paired"]
        if self.decoder_style_ != "mlp":
            assert v_mean_key in mdata.varm.keys()
            assert v_std_key in mdata.varm.keys()
            v_param = (mdata.varm[v_mean_key], mdata.varm[v_std_key])

        # ======================= prepare dataset =======================
        # encode the batch label as integer codes
        merge_obs_from_all_modalities(mdata, self.batch_key_)
        batch_code_key = f"{self.batch_key_}__code"
        mdata.obs[batch_code_key] = self.batch_enc_.transform(
            mdata.obs[self.batch_key_]
        )

        # encode the sslabel as integer codes, nan as -1
        # sslabel_code_key = None  # TODO: cannot use sslable when differential
        # if self.sslabel_key_ is not None:
        #     sslabel_array_key = merge_obs_from_all_modalities(
        #         mdata, self.sslabel_key_
        #     )
        #     sslabel_code_key = f"{self.sslabel_key_}__code"
        #     self.sslabel_enc_ = encode_obs(
        #         mdata, sslabel_array_key, sslabel_code_key, nan_as=-1
        #     )
        # else:
        #     sslabel_code_key = None

        # split data
        label_array = mdata.obs[label_key].values
        label_unique = np.unique(label_array)
        if method == "macro":

            def _iter_pos_neg_dat():
                for labeli in tqdm(
                    label_unique,
                    desc="Differential: ",
                    disable=(self.verbose_ == 0),
                ):
                    # get the positive cells and negative cells
                    mask_pos = label_array == labeli
                    mask_neg = np.logical_not(mask_pos)
                    dat_pos = mdata[mask_pos, :]
                    dat_neg = mdata[mask_neg, :]
                    yield labeli, dat_pos, dat_neg

        else:

            def _iter_pos_neg_dat():
                len_label = len(label_unique)
                for l1, l2 in tqdm(
                    combinations(label_unique, 2),
                    desc="Differential: ",
                    disable=(self.verbose_ == 0),
                    total=int(len_label * (len_label - 1) / 2),
                ):
                    # get the positive cells and negative cells
                    mask_pos = label_array == l1
                    mask_neg = label_array == l2
                    dat_pos = mdata[mask_pos, :]
                    dat_neg = mdata[mask_neg, :]
                    yield f"{l1}-{l2}", dat_pos, dat_neg

        # ======================= calc diff and p =======================
        for label_name, pos_mdat, neg_mdat in _iter_pos_neg_dat():
            loader_pos = get_dataloader(
                mdata=pos_mdat,
                input_key=self.input_key_,
                output_key=self.output_key_,
                batch_key=batch_code_key,
                dlabel_key=self.dlabel_key_,
                # sslabel_code_key, reconstruction do not need label
                sslabel_key=None,
                batch_size=self.batch_size_,
                # shuffle=False,
                num_workers=self.num_workers_,
                pin_memory=self.pin_memory_,
                net_key=None,
                # graph_batch_size=self.graph_batch_size_,
                # drop_self_loop=self.drop_self_loop_,
                # num_negative_samples=self.num_negative_samples_,
                # graph_data_phase="test",
                resample_size=nsamples,
            )
            loader_neg = get_dataloader(
                mdata=neg_mdat,
                input_key=self.input_key_,
                output_key=self.output_key_,
                batch_key=batch_code_key,
                dlabel_key=self.dlabel_key_,
                sslabel_key=None,
                batch_size=self.batch_size_,
                # shuffle=False,
                num_workers=self.num_workers_,
                pin_memory=self.pin_memory_,
                net_key=None,
                # graph_batch_size=self.graph_batch_size_,
                # drop_self_loop=self.drop_self_loop_,
                # num_negative_samples=self.num_negative_samples_,
                # graph_data_phase="test",
                resample_size=nsamples,
            )
            rec_pos = self.trainer_.reconstruct(
                loader_pos,
                v_param=v_param,
                device=self.device_,
                verbose=self.verbose_,
                random=True,
                rec_type=rec_type,
            )
            rec_neg = self.trainer_.reconstruct(
                loader_neg,
                v_param=v_param,
                device=self.device_,
                verbose=self.verbose_,
                random=True,
                rec_type=rec_type,
            )

            # md, pval, bf = [], [], []
            diff = []
            for k in mdata.mod.keys():
                pos_t = rec_pos[k]
                neg_t = rec_neg[k]
                diff.append(pos_t - neg_t)
            diff = torch.cat(diff, dim=1)
            md = diff.mean(dim=0)
            pval = torch.mean(diff > 0.0, dim=0, dtype=torch.float32)
            bf = torch.logit(pval, EPS)

            mdata.var[f"{key_add}_md_{label_name}"] = md.detach().cpu().numpy()
            mdata.var[f"{key_add}_p_{label_name}"] = (
                pval.detach().cpu().numpy()
            )
            mdata.var[f"{key_add}_bf_{label_name}"] = bf.detach().cpu().numpy()

    def save(self, fn: str) -> None:
        torch.save(self.model_.state_dict(), fn)

    def load(self, fn: str, except_prefixs: Optional[Sequence[str]] = None):
        if except_prefixs is None:
            if self.sslabel_key_ is not None:
                except_prefixs = [
                    "encoder.q_c_z",
                    "encoder.q_u_cz",
                    "encoder.p_z_cu",
                    "encoder.diag",
                    "encoder.c_prior",
                ]
            else:
                except_prefixs = []
        state_dict = torch.load(fn, map_location=self.device_)
        new_state_dict = {}
        for k, v in state_dict.items():
            for prefix in except_prefixs:
                if k.statswith(prefix):
                    break
            else:
                new_state_dict[k] = v
        # ms_keys, unex_keys = self.model_.load_state_dict(
        self.model_.load_state_dict(new_state_dict, strict=False)
        return self

# import collections as ct
import collections.abc as cta
import logging
from abc import abstractmethod
from typing import (Any, Dict, Literal, Optional, OrderedDict, Sequence, Tuple,
                    Union)

import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler as lrsch
from torch.utils.data import DataLoader  # Subset,
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..dataset.torch_dataset import (BalanceSizeSampler,
                                     GraphDataLoader, MosaicData,
                                     ParallelDataLoader, SemiSupervisedSampler,
                                     TorchMapDataset)
from ..typehint import BATCH, LOSS  # , EMBED, X_REC
from ..utils import save_args
from .train.accumulator import LossAccumulator
from .train.checkpoint import Checkpointer
from .train.early_stop import EarlyStopper, LearningRateUpdateEarlyStopper
from .train.history import History
from .train.metrics import leiden_cluster
from .train.utils import concat_embeds  # seq2ndarray,; calc_scores
from .train.utils import tensors_to_device, umap_embeds_in_tb


T = torch.Tensor


class MMOModel(nn.Module):
    default_weights: dict[str, float] = {}

    @property
    def arguments(self):
        return self._arguments

    def save(self, fn: str) -> None:
        res = {
            "state_dict": self.state_dict(),
            "arguments": getattr(self, "_arguments", {}),
        }
        torch.save(res, fn)

    @classmethod
    @abstractmethod
    def load(cls, fn: str, **kwargs: dict[str, Any]):
        """载入储存的参数并依次实例化一个相同的模型"""

    @abstractmethod
    def forward(
        self,
        batch: BATCH,
        return_rec_x: bool = False,
        rec_blabel: Optional[T] = None,
    ) -> Dict[str, D.Distribution]:
        """
        返回的是分布。
        batch: 储存有所有输入内容的一个dict；
        return_rec_x：如果是False，则仅返回z；如果是True，则也返回重构的x
        rec_blabel：重构时使用的批次标签，如果是None则使用原始的批次标签
        """

    @abstractmethod
    def step(
        self, batch: BATCH, **kwargs
    ) -> Tuple[
        # TODO: 其实下面的这输出只适用于mmAAVI，如果想要适用于更广泛的模型（比如
        #   非encoder-decoder架构），则需要再进行修改
        Dict[str, Any],  # enc_res
        Dict[str, Any],  # dec_res
        Dict[str, Any],  # disc_res
        LOSS,
    ]:
        """
        接受样本并最终计算得到loss的过程，返回latent variable和losses
        其中loss是一个dict，其中必须有total和metric两个元素，total用于计算梯度，
        metric用于进行评价
        """

    def config_optim(
        self,
        lr: float,
        optimizer: str = "adam",
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-5,
        lr_schedual: Optional[str] = None,
        sch_kwargs: dict[str, Any] = {},
    ) -> Tuple[optim.Optimizer, Optional[lrsch._LRScheduler]]:
        if optimizer == "adam":
            opt = optim.Adam(
                self.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        elif optimizer == "rmsprop":
            opt = optim.RMSprop(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )

        if lr_schedual is None:
            return opt, None

        if lr_schedual == "reduce_on_plateau":
            schedual = lrsch.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=sch_kwargs.get("factor", 0.1),
                patience=sch_kwargs.get("patience", 10),
                threshold=sch_kwargs.get("threshold", 0.0),
                threshold_mode=sch_kwargs.get("threshold_mode", "abs"),
                min_lr=sch_kwargs.get("min_lr", 0.0),
            )
        elif lr_schedual == "exp":
            schedual = lrsch.ExponentialLR(opt, sch_kwargs["gamma"])
        elif lr_schedual == "multi_step":
            schedual = lrsch.MultiStepLR(
                opt,
                milestones=sch_kwargs["milestones"],
                gamma=sch_kwargs["gamma"],
            )
        else:
            raise NotImplementedError
        return opt, schedual

    @classmethod
    def configure_data_to_loader(
        cls,
        data: MosaicData,
        phase: str = "train",
        shuffle: bool = True,
        batch_size: int = 128,
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
        ] = None,
        net_use: Optional[str] = None,
        net_style: Literal["dict", "sarr", "walk"] = "dict",
        net_batch_size: Union[int, float] = 0.5,
        drop_self_loop: bool = True,
        walk_length: int = 20,
        context_size: int = 10,
        walks_per_node: int = 10,
        num_negative_samples: int = 1,
        p: float = 1.0,
        q: float = 1.0,
    ) -> Tuple[
        DataLoader,  # dataloader
        OrderedDict[str, int],  # input_dims
        OrderedDict[str, int],  # output_dims
    ]:
        """
        sslabel_codes是用来保证train、valid、test拥有相同的codes
        如果phase=test，则仅返回torch dataset部分，并且shuffle、drop_last、resample等参数
        并不起效果。phase=train和valid是相同的效果
        """
        # ------------------------------------
        # Omics数据部分
        # ------------------------------------
        # 创建torch style mapping dataset
        dataset_omic = TorchMapDataset(
            data,
            input_use=input_use,
            output_use=output_use,
            obs_blabel=obs_blabel,
            obs_dlabel=obs_dlabel,
            obs_sslabel=obs_sslabel,
            obs_sslabel_full=obs_sslabel_full,
            impute_miss=impute_miss,
            sslabel_codes=sslabel_codes,
        )
        inpt_dims = dataset_omic._inpt_grid.omics_dims_dict
        oupt_dims = dataset_omic._oupt_grid.omics_dims_dict

        if phase == "test":
            loader_omic = DataLoader(
                dataset_omic,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                sampler=None,
                collate_fn=dataset_omic.get_collated_fn(),
                drop_last=False,
            )
            return loader_omic, inpt_dims, oupt_dims

        # calculate resample ratio ...
        if resample is not None:
            if resample == "max":
                stand = max(data.batch_dims)
                resample_ratio = {
                    k: stand / n for k, n in data.batch_dims_dict.items()
                }
            elif resample == "min":
                stand = min(data.batch_dims)
                resample_ratio = {
                    k: stand / n for k, n in data.batch_dims_dict.items()
                }
            else:
                resample_ratio = {
                    k: resample.get(k, 1.0) for k in data.batch_names
                }
        else:
            resample_ratio = None

        if obs_sslabel is not None:
            is_labeled = data.obs[obs_sslabel].notna().values
            ss_sampler = SemiSupervisedSampler(
                is_labeled,
                batch_size,
                ss_batch_size,
                resample_size=resample_ratio,
                blabels=dataset_omic._blabel.values,
                drop_last=drop_last,
            )
            loader_omic = DataLoader(
                dataset_omic,
                batch_sampler=ss_sampler,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=dataset_omic.get_collated_fn(),
                # drop_last=drop_last,
            )
        else:
            if resample_ratio is not None:
                sampler = BalanceSizeSampler(
                    resample_size=resample_ratio,
                    blabels=dataset_omic._blabel.values,
                )
                shuffle = None
            else:
                sampler = None
            loader_omic = DataLoader(
                dataset_omic,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler,
                collate_fn=dataset_omic.get_collated_fn(),
                drop_last=drop_last,
            )

        if net_use is None:
            return loader_omic, inpt_dims, oupt_dims

        # ------------------------------------
        # graph数据部分
        # ------------------------------------

        if net_use:
            assert (
                output_use is None
            ), "now only support using net when output counts(X)"
            net = data.nets[net_use]
            if net_style == "dict":
                net.as_sparse_type("coo", only_sparse=False)
                nets = {}
                for o1, o2, dati in net.items():
                    if dati is None:
                        continue
                    # logging.info("  %s x %s ..." % (o1, o2))
                    nets[(o1, o2)] = dati
            elif net_style in ["sarr", "walk"]:
                net.as_sparse_type("csr", only_sparse=False)
                nets = net.to_array(True)
                nets = nets.tocoo()
                # 塑造对称性, 加入一个自回路径
                i, j, v = nets.row, nets.col, nets.data
                new_i = np.concatenate([i, j, np.arange(nets.shape[0])])
                new_j = np.concatenate([j, i, np.arange(nets.shape[1])])
                new_v = np.concatenate([v, v, np.ones(nets.shape[0])])
                new_s = 2 * (new_v > 0.0).astype(float) - 1  # NOTE: 1和-1
                new_v = np.abs(new_v)
                nets = (nets.shape, new_i, new_j, new_s, new_v)
            else:
                raise NotImplementedError

        if isinstance(net_batch_size, float):
            assert net_batch_size > 0.0 and net_batch_size <= 1.0

        graph_loader = GraphDataLoader(
            nets,
            net_style,
            batch_size=net_batch_size,
            drop_self_loop=drop_self_loop,
            num_workers=num_workers,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=p,
            q=q,
        )
        return (
            ParallelDataLoader(
                loader_omic, graph_loader, cycle_flags=[False, True]
            ),
            inpt_dims,
            oupt_dims,
        )

    def _handle_weight_values(self, max_epochs: int, kweights: dict):
        # 训练时使用的权重
        self.weight_values = {}
        for k, v in self.default_weights.items():
            v = kweights.get(k, v)
            if isinstance(v, (float, int)):
                logging.info("weight values: %s is %.2f" % (k, v))
                self.weight_values[k] = np.full(max_epochs, v)
            else:
                mid_ind = len(v) // 2
                logging.info(
                    "weight values: %s is %.2f(0) -> %.2f(%d) -> %.2f(%d)"
                    % (k, v[0], v[mid_ind], mid_ind, v[-1], (len(v) - 1))
                )
                if len(v) < max_epochs:
                    v = list(v)
                    v += [v[-1]] * (max_epochs - len(v))
                self.weight_values[k] = v

    @save_args(exclude=("train_loader", "valid_loader", "tensorboard_dir"))
    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
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
        valid_umap_interval: int = 3,
        valid_show_umap: Optional[Union[str, Sequence[str]]] = None,
        checkpoint_best: bool = True,
        early_stop: bool = True,
        early_stop_patient: int = 10,
        tensorboard_dir: Optional[str] = None,
        verbose: int = 2,
        **kweights: Union[float, Sequence, np.ndarray],
    ) -> Tuple[dict[str, pd.DataFrame], dict]:
        """
        valid_show_umap:
            如果是None，则不绘制umap；
            如果是str或者seq of str，则以此为categories来绘制，每个str绘制
                一个子图，这些str是mdata.obs中的。
        verbose:
            0: show nothing
            1: show epoch bar
            2: show batch bar
            3: print early_stop
            4: print all step information
        """

        assert optimizer in ["adam", "rmsprop"]

        # simulate the 2-step for testing the timing
        # self._flag_no_grl = no_grl

        for k, v in kweights.items():
            assert (
                k in self.default_weights
            ), "%s is not a legal weight, legal weights are %s" % (
                k,
                ",".join(self.default_weights.keys()),
            )
            if isinstance(v, (cta.Sequence, np.ndarray)):
                assert len(v) <= max_epochs, (
                    "the length of weight sequence %s is larger to max_epochs"
                    % k
                )

        if valid_loader is not None:
            if valid_show_umap is not None:
                if isinstance(valid_show_umap, str):
                    valid_show_umap = [valid_show_umap]
                if isinstance(valid_loader, ParallelDataLoader):
                    omic_loader = valid_loader.data_loaders[0]
                else:
                    omic_loader = valid_loader
                valid_data = omic_loader.dataset._mdata
                obs_labels = {
                    namei: valid_data.obs.loc[:, namei]  # 这里是series
                    for namei in valid_show_umap
                }

        device = torch.device(device)
        self.to(device)

        writer = SummaryWriter(tensorboard_dir) if tensorboard_dir else None
        accumutor = LossAccumulator()
        history_recoder = History()
        if checkpoint_best:
            checkpointer = Checkpointer("metric", "valid")
        if early_stop:
            early_stopper = EarlyStopper("metric", "valid", early_stop_patient)
        # 记录上一次的lr，以及lr变化的次数
        flag_es_lr = (
            lr_schedual is not None
            and early_stop
            and sch_max_update is not None
        )
        if flag_es_lr:
            early_stopper_lr = LearningRateUpdateEarlyStopper(
                learning_rate, sch_max_update
            )

        opt, schedual = self.config_optim(
            lr=learning_rate,
            optimizer=optimizer,
            betas=betas,
            weight_decay=weight_decay,
            lr_schedual=lr_schedual,
            sch_kwargs=sch_kwargs,
        )

        flag_cat_valid_outputs = valid_show_umap and writer is not None

        self._handle_weight_values(max_epochs, kweights)

        for e in tqdm(
            range(max_epochs), desc="Epoch: ", disable=(verbose < 1)
        ):
            self._e = e
            # 取出当前批次使用的权重
            self.weight_values_e = {
                k: v[e] for k, v in self.weight_values.items()
            }
            if writer is not None:
                # 记录lr的变化
                # 这里所有的参数使用相同lr，选择第一个即可
                current_lr = opt.param_groups[0]["lr"]
                writer.add_scalar("lambda/learning_rate", current_lr, e)
                # 将这些权重加入到tensorboard中
                for k, v in self.weight_values_e.items():
                    writer.add_scalar("lambda/%s" % k, v, e)

            accumutor.init()
            self.train()
            self._phase = "train"
            with torch.enable_grad():
                for batch in tqdm(
                    train_loader,
                    desc="Train Batch: ",
                    leave=False,
                    disable=(verbose < 2),
                ):
                    batch = tensors_to_device(batch, device)
                    losses = self.step(batch)[-1]
                    opt.zero_grad()
                    losses["total"].backward()
                    if grad_clip is not None:
                        nn.utils.clip_grad.clip_grad_norm_(
                            self.parameters(), grad_clip
                        )
                    opt.step()

                    accumutor.add(**losses)

            losses_tr = accumutor.calc()
            history_recoder.append(e, losses_tr, "train", log_writer=writer)
            if verbose >= 4:
                tqdm.write(history_recoder.show_record("train", -1))

            # if (
            #     (valid_loader is not None) and
            #     ((e % valid_interval == 0) or e == (max_epochs - 1))
            # ):
            if valid_loader is not None:
                accumutor.init()
                self.eval()
                self._phase = "valid"
                with torch.no_grad():
                    # outputs, labels, blabels = [], [], []
                    outputs, indexes = [], []
                    for batch in tqdm(
                        valid_loader,
                        desc="Valid Batch: ",
                        leave=False,
                        disable=(verbose < 2),
                    ):
                        batch = tensors_to_device(batch, device)
                        enc_res, _, _, losses = self.step(batch)
                        accumutor.add(**losses)

                        if flag_cat_valid_outputs:
                            # 记录embed用于umap绘图
                            embed = [enc_res["z"].mean]
                            if "c" in enc_res:
                                embed.append(enc_res["c"].probs)
                            outputs.append(embed)
                            indexes += batch["ind"]

                losses_va = accumutor.calc()
                history_recoder.append(
                    e, losses_va, "valid", log_writer=writer
                )
                if verbose >= 4:
                    tqdm.write(history_recoder.show_record("valid", -1))

                if flag_cat_valid_outputs:
                    # labels = np.concatenate(labels)
                    # blabels = torch.cat(blabels).detach().cpu().numpy()
                    z, clu_prob = concat_embeds(outputs)

                    if clu_prob is None:
                        # 需要通过leiden cluster来得到clu
                        clu = leiden_cluster(
                            z.detach().cpu().numpy(), resolution=0.2, n_jobs=4
                        )
                    else:
                        clu = clu_prob.argmax(dim=1).detach().cpu().numpy()

                if (
                    ((e % valid_umap_interval == 0) or e == (max_epochs - 1))
                    and valid_show_umap
                    and writer is not None
                ):
                    # 绘制umap到tensorboard中
                    # valid可能也是shuffle的，则需要通过index来重新得到这些
                    # label的顺序
                    obs_labels_i = {
                        k: v.loc[indexes].values for k, v in obs_labels.items()
                    }
                    obs_labels_i["cluster"] = clu
                    umap_embeds_in_tb(e, writer, z, obs_labels_i)

            # 监控并保存模型
            # 这个score_main可能lr_sch和save_best都会用到
            if lr_schedual == "reduce_on_plateau" or checkpoint_best:
                score_main = checkpointer.watch(history_recoder)

            # 如果需要保存最好的模型而非最后，需要根据选择出的score进行更新
            if checkpoint_best:
                # 需要先运行watch方法，才能运行update_best
                checkpointer.update_best(self, verbose)

            # 更新学习率，可能需要score的参与
            if schedual is not None:
                if (
                    isinstance(schedual, lrsch.ReduceLROnPlateau)
                    and valid_loader is not None
                ):
                    schedual.step(score_main)
                else:
                    schedual.step()

            # 根据当前的结果决定是否早停
            if early_stop:
                early_stopper.watch(history_recoder)
                if early_stopper.is_stop():
                    break
            # 根据当前的lr更新决定是否早停
            if flag_es_lr:
                early_stopper_lr.watch(opt)
                if early_stopper_lr.is_stop():
                    break

        if verbose >= 3:
            if early_stop:
                early_stopper.print_msg()
            if flag_es_lr:
                early_stopper_lr.print_msg()

        best = None
        if checkpoint_best:
            best = checkpointer._best
            if verbose >= 3:
                tqdm.write(
                    "best model at epoch %d, best score is %.4f"
                    % (best["epoch"], best["value"])
                )
            checkpointer.apply_state_dict(self)

        hist_dfs = history_recoder.to_dfs()
        return hist_dfs, best

    def encode(
        self,
        loader: DataLoader,
        device: str = "cuda:0",
        verbose: int = 1,
    ) -> Dict[str, T]:
        device = torch.device(device)
        self.to(device)

        self.eval()
        with torch.no_grad():
            outputs = []
            for batch in tqdm(loader, desc="Predict: ", disable=verbose < 1):
                batch = tensors_to_device(batch, device)
                embed = self(batch, return_rec_x=False)
                outputs.append(embed)

        z = torch.cat([di["z"].mean for di in outputs], dim=0)
        if "c" in outputs[0]:
            c = torch.cat([di["c"].probs for di in outputs], dim=0)

        return {"z": z, "c": c}

    def reconstruct(self) -> OrderedDict[str, T]:
        raise NotImplementedError

from typing import Union, Optional, Tuple, Any, Dict, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler as lrsch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from tqdm import tqdm

from .dataset import ParallelDataLoader
from .network import MMAAVINET
from .train_utils.accumulator import LossAccumulator
from .train_utils.checkpoint import Checkpointer
from .train_utils.early_stop import (
    EarlyStopper,
    LearningRateUpdateEarlyStopper,
)
from .train_utils.history import History

# from .train_utils.metrics import leiden_cluster
from .train_utils.utils import tensors_to_device  # , concat_embeds,


T = torch.Tensor
LOADER = Union[DataLoader, ParallelDataLoader]


class Trainer:
    def __init__(self, model: MMAAVINET) -> None:
        self.model = model

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
                self.model.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        elif optimizer == "rmsprop":
            opt = optim.RMSprop(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
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

    def train(
        self,
        train_loader: LOADER,
        valid_loader: Optional[LOADER] = None,
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
        # **kweights: Union[float, Sequence, np.ndarray],
    ) -> Tuple[dict[str, pd.DataFrame], dict]:
        assert optimizer in ["adam", "rmsprop"]

        # simulate the 2-step for testing the timing
        # self._flag_no_grl = no_grl

        # for k, v in kweights.items():
        #     assert (
        #         k in self.default_weights
        #     ), "%s is not a legal weight, legal weights are %s" % (
        #         k,
        #         ",".join(self.default_weights.keys()),
        #     )
        #     if isinstance(v, (cta.Sequence, np.ndarray)):
        #         assert len(v) <= max_epochs, (
        #             "the length of weight sequence %s is "
        #                  "larger to max_epochs"
        #             % k
        #         )

        if valid_loader is not None:
            pass
            # if valid_show_umap is not None:
            #     if isinstance(valid_show_umap, str):
            #         valid_show_umap = [valid_show_umap]
            #     if isinstance(valid_loader, ParallelDataLoader):
            #         omic_loader = valid_loader.data_loaders[0]
            #     else:
            #         omic_loader = valid_loader
            #     valid_data = omic_loader.dataset._mdata
            #     obs_labels = {
            #         namei: valid_data.obs.loc[:, namei]  # 这里是series
            #         for namei in valid_show_umap
            #     }
            #
        device = torch.device(device)
        self.model.to(device)

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

        # flag_cat_valid_outputs = valid_show_umap and writer is not None
        flag_cat_valid_outputs = writer is not None

        # self._handle_weight_values(max_epochs, kweights)

        for e in tqdm(
            range(max_epochs), desc="Epoch: ", disable=(verbose < 1)
        ):
            self._e = e
            # 取出当前批次使用的权重
            # self.weight_values_e = {
            #     k: v[e] for k, v in self.weight_values.items()
            # }
            if writer is not None:
                # 记录lr的变化
                # 这里所有的参数使用相同lr，选择第一个即可
                current_lr = opt.param_groups[0]["lr"]
                writer.add_scalar("lambda/learning_rate", current_lr, e)
                # 将这些权重加入到tensorboard中
                for k, v in self.weight_values_e.items():
                    writer.add_scalar("lambda/%s" % k, v, e)

            accumutor.init()
            self.model.train()
            self._phase = "train"
            with torch.enable_grad():
                for batch in tqdm(
                    train_loader,
                    desc="Train Batch: ",
                    leave=False,
                    disable=(verbose < 2),
                ):
                    batch = tensors_to_device(batch, device)
                    losses = self.model.step(batch)[-1]
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
                self.model.eval()
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
                        enc_res, _, _, losses = self.model.step(batch)
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
                    # TODO: 可以实时计算一些指标
                    pass
                    # labels = np.concatenate(labels)
                    # # blabels = torch.cat(blabels).detach().cpu().numpy()
                    # z, clu_prob = concat_embeds(outputs)

                    # if clu_prob is None:
                    #     # 需要通过leiden cluster来得到clu
                    #     clu = leiden_cluster(
                    #         z.detach().cpu().numpy(),
                    #         resolution=0.2, n_jobs=4
                    #     )
                    # else:
                    #     clu = clu_prob.argmax(dim=1).detach().cpu().numpy()

                # if (
                #     ((e % valid_umap_interval == 0) or e == (max_epochs - 1))
                #     and valid_show_umap
                #     and writer is not None
                # ):
                #     # 绘制umap到tensorboard中
                #     # valid可能也是shuffle的，则需要通过index来重新得到这些
                #     # label的顺序
                #     obs_labels_i = {
                #         k: v.loc[indexes].values
                #         for k, v in obs_labels.items()
                #     }
                #     obs_labels_i["cluster"] = clu
                #     umap_embeds_in_tb(e, writer, z, obs_labels_i)

            # 监控并保存模型
            # 这个score_main可能lr_sch和save_best都会用到
            if lr_schedual == "reduce_on_plateau" or checkpoint_best:
                score_main = checkpointer.watch(history_recoder)

            # 如果需要保存最好的模型而非最后，需要根据选择出的score进行更新
            if checkpoint_best:
                # 需要先运行watch方法，才能运行update_best
                checkpointer.update_best(self.model, verbose)

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
            checkpointer.apply_state_dict(self.model)

        hist_dfs = history_recoder.to_dfs()
        return hist_dfs, best

    def encode(
        self,
        loader: LOADER,
        device: str = "cuda:0",
        verbose: int = 1,
    ) -> Dict[str, T]:
        device = torch.device(device)
        self.model.to(device)

        self.model.eval()
        with torch.no_grad():
            outputs = []
            for batch in tqdm(loader, desc="Encoding: ", disable=verbose < 1):
                batch = tensors_to_device(batch, device)
                embed = self.model(batch)
                outputs.append(embed)

        res = {"z": torch.cat([di["z"].mean for di in outputs], dim=0)}
        if "att" in outputs[0]:
            res["att"] = torch.cat([di["att"] for di in outputs], dim=0)
        if "c" in outputs[0]:
            res["c"] = torch.cat([di["c"].probs for di in outputs], dim=0)
        return res

    def reconstruct(
        self,
        loader: LOADER,
        v_param: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        device: str = "cuda:0",
        verbose: int = 1,
        random: bool = True,
        rec_type: Literal["mean", "sample"] = "mean",
    ) -> Dict[str, T]:
        device = torch.device(device)
        self.model.to(device)

        if v_param is None:
            v_dist = None
        else:
            v_mean, v_std = v_param
            v_mean = torch.tensor(v_mean, dtype=torch.float32, device=device)
            v_std = torch.tensor(v_std, dtype=torch.float32, device=device)
            v_dist = Normal(v_mean, v_std)

        res = {}
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(
                loader, desc="Reconstruct: ", disable=verbose < 1,
                leave=False
            ):
                batch = tensors_to_device(batch, device)
                recon = self.model.reconstruct(
                    batch, v_dist=v_dist, random=random
                )
                for k, d in recon.items():
                    res.setdefault(k, []).append(
                        d.mean if rec_type == "mean" else d.sample()
                    )
        res = {k: torch.cat(v, dim=0) for k, v in res.items()}
        return res

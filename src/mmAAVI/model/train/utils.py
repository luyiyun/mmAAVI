from collections import OrderedDict as odict
from typing import Any, Dict, Optional, Sequence, Tuple

import colorcet as cc
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from cuml.manifold.umap import UMAP
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             normalized_mutual_info_score, roc_auc_score)
from torch.utils.tensorboard import SummaryWriter
from matplotlib.patches import Rectangle

# from ..model import ModelBase
from ...typehint import BATCH, EMBED
from .metrics import (entropy_for_distant_labels, entropy_for_predict_proba,
                      graph_connectivity, kBET, leiden_cluster)
from .weighter import Weighter

T = torch.Tensor


def onehot(arr: np.ndarray, n: int) -> np.ndarray:
    return np.eye(n)[arr.astype(int)]


def seq2ndarray(seq: Sequence) -> np.ndarray:
    if isinstance(seq, np.ndarray):
        return seq
    elif isinstance(seq, list):
        return np.array(seq)
    elif torch.is_tensor(seq):
        return seq.detach().cpu().numpy()
    else:
        raise NotImplementedError


def tensors_to_device(tensors: BATCH, device: torch.device) -> BATCH:
    if torch.is_tensor(tensors):
        return tensors.to(device)
    elif isinstance(tensors, str):
        return tensors
    elif isinstance(tensors, dict):
        return {k: tensors_to_device(v, device) for k, v in tensors.items()}
    elif isinstance(tensors, odict):
        return odict(
            [(k, tensors_to_device(v, device)) for k, v in tensors.items()]
        )
    elif isinstance(tensors, Sequence):
        return [tensors_to_device(t, device) for t in tensors]
    else:
        return tensors
        # raise ValueError("unknown type: %s" % (str(type(tensors))))


def concat_embeds(embeds: Sequence[EMBED]) -> Tuple[T, Optional[T]]:
    # 得到聚类和z
    if torch.is_tensor(embeds[0]):
        # 这时候只进行了降维，只有z，所以接下来的聚类要使用kmeans进行
        z = torch.cat(embeds, dim=0)
        clu_prob = None
    elif isinstance(embeds[0], (list, tuple)) and len(embeds[0]) >= 2:
        # 这时候第一个是z，第二个是聚类标签
        z, clu_prob = list(zip(*[output[:2] for output in embeds]))
        z = torch.cat(z, dim=0)
        clu_prob = torch.cat(clu_prob, dim=0)
    else:
        raise NotImplementedError
    return z, clu_prob


def plot_umap_ax(
    ax: Any,
    u1: np.ndarray,
    u2: np.ndarray,
    hue: np.ndarray,
    s: Optional[float] = None,
    alpha: float = 1,
    title: Optional[str] = None,
):
    # 如果出现np.NaN，则np.unique会报错，这里将缺失值全部去掉
    mask = pd.notna(hue)
    hue = hue[mask]
    u1, u2 = u1[mask], u2[mask]

    hue_uni = np.unique(hue)
    palette = (
        sns.color_palette()
        if len(hue_uni) <= 10
        else sns.color_palette(cc.glasbey, n_colors=len(hue_uni))
    )
    s = s if s is not None else min(10, 5000 / u1.shape[0])
    for i, huei in enumerate(hue_uni):
        u1i = u1[mask := (hue == huei)]
        u2i = u2[mask]
        ax.plot(
            u1i,
            u2i,
            ".",
            markersize=s,
            color=palette[i],
            label=huei,
            alpha=alpha,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    if title is not None:
        ax.set_title(title)
    return ax


def plot_umap(
    u1: np.ndarray,
    u2: np.ndarray,
    categories: Sequence[np.ndarray],
    categories_name: Optional[Sequence[str]] = None,
    s: Optional[float] = None,
    alpha: float = 1,
):
    if categories_name is not None:
        assert len(categories) == len(categories_name)
    # s = s if s is not None else 8000 / u1.shape[0]
    # ss = 10 / s
    nsub = len(categories)
    nr = int(np.ceil(np.sqrt(nsub)))
    nc = int(np.ceil(nsub / nr))
    fig, axs = plt.subplots(
        nrows=nr,
        ncols=nc,
        figsize=(nc * 4, nr * 4),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axs = axs.flatten()
    for i, catei in enumerate(categories):
        ax = axs[i]
        namei = categories_name[i] if categories_name is not None else None
        plot_umap_ax(ax, u1, u2, catei, s, alpha=alpha, title=namei)

    handles, labels = [], []
    for ax, namei in zip(axs, categories_name):
        handles.append(Rectangle((0, 0), 0, 0, color='w'))  # a placeholder
        if namei.startswith("_"):
            namei = namei[1:]
        labels.append(namei if namei is not None else " ")

        handlei, labeli = ax.get_legend_handles_labels()
        handles.extend(handlei)
        labels.extend(labeli)
    labels = [li if len(li) <= 10 else li[:10] for li in labels]
    leg = fig.legend(
        handles,
        labels,
        # 在layout=constrained的前提下，可以直接使用outside。
        loc="outside lower center",
        # 使用upper center+bbox的组合，figure中的legend会被截断
        # loc="upper center",
        # bbox_to_anchor=(0.0, -0.4, 1.0, 0.5),
        frameon=False,
        fancybox=False,
        ncols=9,
        columnspacing=0.2,
        handletextpad=0.1,
    )
    for h in leg.legend_handles:
        if not isinstance(h, Rectangle):
            h.set_markersize(10.0)
    leg.set_in_layout(True)

    # fig.subplots_adjust(top=1.0, bottom=0.1)

    return fig, axs


def umap_embeds_in_tb(
    epoch: int,
    writer: SummaryWriter,
    z: torch.Tensor,
    obs_labels: dict[str, np.ndarray],
) -> None:
    # umap可视化
    # ---------------------------------------------------------------------
    # 使用cuml
    # cuml无法通过参数指定使用的devices，所以我们直接将pytorch使用的device
    #   也固定为第一个gpu，然后通过环境变量CUDA_VISIBLE_DEVICES来指定训练时
    #   使用哪个gpu，这样可以保证cuml和pytorch使用一个gpu
    umap_op = UMAP(n_components=2, random_state=0)
    x_umap = umap_op.fit_transform(z)
    x_umap = cp.asnumpy(x_umap)
    # ---------------------------------------------------------------------
    # 使用umap(cpu)
    # z, clu = z.cpu().numpy(), clu.cpu().numpy()
    # umap_op = UMAP(
    #     n_components=2, n_neighbors=30, min_dist=0.2, random_state=0,
    #     init="random"  # init使用random后会提高性能
    # )
    # x_umap = umap_op.fit_transform(z)
    # ---------------------------------------------------------------------
    fig, _ = plot_umap(
        x_umap[:, 0],
        x_umap[:, 1],
        list(obs_labels.values()),
        list(obs_labels.keys()),
        alpha=0.8,
    )
    # x_umap = pd.DataFrame(x_umap, columns=["Z1", "Z2"])
    # for k, v in obs_labels.items():
    #     x_umap[k] = pd.Categorical(v)
    # nsub = len(obs_labels)
    # nr = int(np.ceil(np.sqrt(nsub)))
    # nc = int(np.ceil(nsub / nr))
    # fig, axs = plt.subplots(
    #     nrows=nr, ncols=nc, figsize=(nc * 5, nr * 5),
    #     sharex=True, sharey=True
    # )
    # axs = axs.flatten()
    #
    # for i, k in enumerate(obs_labels.keys()):
    #     ncells = pd.notnull(obs_labels[k]).sum()
    #     sns.scatterplot(
    #         data=x_umap,
    #         x="Z1",
    #         y="Z2",
    #         hue=k,
    #         s=30000 / ncells,
    #         ax=axs[i],
    #         legend=True,
    #         alpha=0.5,
    #     )
    #     axs[i].set_title(k)
    #
    # fig.tight_layout()
    writer.add_figure("umap", fig, epoch)
    writer.flush()
    plt.close("all")
    return


def calc_scores(
    z: torch.Tensor,
    clu: np.ndarray,
    batches: np.ndarray,
    labels: Optional[np.ndarray] = None,
    clu_prob: Optional[torch.Tensor] = None,
    sslabels: Optional[np.ndarray] = None,
    sslabels_full: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if sslabels is not None or sslabels_full is not None:
        assert (
            clu_prob is not None
        ), "must given clu_prob when evaluate semi-supervised"

    # 计算指标
    scores = {}
    nclu = np.unique(clu).shape[0]
    scores["entropy"] = entropy_for_distant_labels(clu, nclu, True)
    if clu_prob is not None:
        scores["centropy"] = (
            entropy_for_predict_proba(clu_prob, True).mean().item()
        )
    z = z.cpu().numpy() if torch.is_tensor(z) else cp.asnumpy(z)
    scores["kBET"] = kBET(batch=batches, data=z, K=30, alpha=0.05, n_jobs=5)[0]
    if labels is not None:
        scores["gc"] = graph_connectivity(X=z, groups=labels, k=30, n_jobs=5)
        scores["ari"] = adjusted_rand_score(labels, clu)
        scores["nmi"] = normalized_mutual_info_score(labels, clu)

    if sslabels_full is not None:
        pred_prob = clu_prob.detach().cpu().numpy()
        pred = pred_prob.argmax(axis=1)
        sslabels_f_oh = onehot(sslabels_full, pred_prob.shape[1])
        batch_codes = np.unique(batches)
        # 计算一个full acc、auc，各个batch的acc、auc
        scores["acc_full"] = accuracy_score(sslabels_full, pred)
        scores["auc_full"] = roc_auc_score(sslabels_f_oh, pred_prob)
        for bi in batch_codes:
            bi_mask = batches == bi
            scores["acc_batch%d" % bi] = accuracy_score(
                sslabels_full[bi_mask], pred[bi_mask]
            )
            scores["auc_batch%d" % bi] = roc_auc_score(
                sslabels_f_oh[bi_mask], pred_prob[bi_mask]
            )
        if sslabels is not None:
            # 计算已知标签的acc、auc，未知标签的acc、auc
            ss_mask = sslabels == -1
            scores["acc_unknow"] = accuracy_score(
                sslabels_full[ss_mask], pred[ss_mask]
            )
            scores["auc_unknow"] = roc_auc_score(
                sslabels_f_oh[ss_mask], pred_prob[ss_mask]
            )
            ss_mask = ~ss_mask
            scores["acc_know"] = accuracy_score(
                sslabels_full[ss_mask], pred[ss_mask]
            )
            scores["auc_know"] = roc_auc_score(
                sslabels_f_oh[ss_mask], pred_prob[ss_mask]
            )

    return scores


def visual_and_calc_scores(
    outputs: Sequence[EMBED],
    labels: np.ndarray,
    blabels: np.ndarray,
    epoch: int,
    leiden_kwargs: Optional[Dict[str, Any]] = None,
    writer: Optional[SummaryWriter] = None,
    use_leiden: bool = False,
) -> Dict[str, float]:
    # 得到聚类和z
    if torch.is_tensor(outputs[0]):
        # 这时候只进行了降维，只有z，所以接下来的聚类要使用kmeans进行
        z = torch.cat(outputs, dim=0)
        clu_prob = None
    elif use_leiden:
        z = torch.cat([out[0] for out in outputs], dim=0)
        clu_prob = None
    elif isinstance(outputs[0], (list, tuple)) and len(outputs[0]) >= 2:
        # 这时候第一个是z，第二个是聚类标签
        z, clu_prob = list(zip(*[output[:2] for output in outputs]))
        z = torch.cat(z, dim=0)
        clu_prob = torch.cat(clu_prob, dim=0)
    else:
        raise NotImplementedError

    # cluster
    if clu_prob is not None:
        clu = clu_prob.argmax(dim=1).detach().cpu().numpy()
    else:
        clu = leiden_cluster(X=z.detach().cpu().numpy(), **leiden_kwargs)

    if writer is not None:
        # umap可视化
        # ---------------------------------------------------------------------
        # 使用cuml
        # cuml无法通过参数指定使用的devices，所以我们直接将pytorch使用的device
        #   也固定为第一个gpu，然后通过环境变量CUDA_VISIBLE_DEVICES来指定训练时
        #   使用哪个gpu，这样可以保证cuml和pytorch使用一个gpu
        umap_op = UMAP(n_components=2, random_state=0)
        x_umap = umap_op.fit_transform(z)
        x_umap, clu = cp.asnumpy(x_umap), cp.asnumpy(clu)
        # ---------------------------------------------------------------------
        # 使用umap(cpu)
        # z, clu = z.cpu().numpy(), clu.cpu().numpy()
        # umap_op = UMAP(
        #     n_components=2, n_neighbors=30, min_dist=0.2, random_state=0,
        #     init="random"  # init使用random后会提高性能
        # )
        # x_umap = umap_op.fit_transform(z)
        # ---------------------------------------------------------------------
        x_umap = pd.DataFrame(x_umap, columns=["Z1", "Z2"])
        for name, arr in zip(
            ["batch", "label", "cluster"], [blabels, labels, clu]
        ):
            x_umap[name] = arr
            x_umap[name] = x_umap[name].astype("category")
        fg_label = sns.relplot(
            data=x_umap,
            x="Z1",
            y="Z2",
            hue="label",
            col="batch",
            col_wrap=2,
            s=5,
            height=4,
        )
        fg_cluster = sns.relplot(
            data=x_umap,
            x="Z1",
            y="Z2",
            hue="cluster",
            col="batch",
            col_wrap=2,
            s=5,
            height=4,
        )
        writer.add_figure("umap/label", fg_label.figure, epoch)
        writer.add_figure("umap/cluster", fg_cluster.figure, epoch)
        # 将画图产生的memory清除
        fg_label.figure.clear()
        fg_cluster.figure.clear()
        plt.close("all")
        writer.flush()

    # 计算指标
    scores = {}
    nclu = np.unique(clu).shape[0]
    scores["entropy"] = entropy_for_distant_labels(clu, nclu, True)
    if clu_prob is not None:
        scores["centropy"] = (
            entropy_for_predict_proba(clu_prob, True).mean().item()
        )
    z = z.cpu().numpy() if torch.is_tensor(z) else cp.asnumpy(z)
    scores["gc"] = graph_connectivity(X=z, groups=labels, k=30, n_jobs=5)
    scores["kBET"] = kBET(batch=blabels, data=z, K=30, alpha=0.05, n_jobs=5)[0]
    scores["ari"] = adjusted_rand_score(labels, clu)
    scores["nmi"] = normalized_mutual_info_score(labels, clu)

    if writer is not None:
        for k, v in scores.items():
            writer.add_scalar("score/%s" % k, v, epoch)

    return scores


def append_hist_and_tensorboard(
    epoch: int,
    scores: Dict[str, float],
    hist: Dict[str, list],
    phase: str = "train",
    writer: Optional[SummaryWriter] = None,
) -> None:
    hist["epoch"].append(epoch)
    for k, v in scores.items():
        hist.setdefault(k, []).append(v)
    if writer is not None:
        for k, v in scores.items():
            writer.add_scalar("%s/%s" % (phase, k), v, epoch)
        writer.flush()


def sum_losses(epoch: int, losses: Dict[str, T], weighter: Weighter) -> T:
    loss = 0.0
    for k, v in losses.items():
        if k in weighter:
            loss += v * weighter.at(k, epoch)
        else:
            loss += v
    return loss

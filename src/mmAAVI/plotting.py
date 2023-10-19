from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text


def volcano(
    bf: np.ndarray,
    md: np.ndarray,
    colors: Optional[np.ndarray] = None,
    anno: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    x_thre: Optional[float] = 2,
    y_thre: Optional[float] = 2,
    ax: Optional[plt.Axes] = None,
    s: float = 10,
    fontsize: float = 10,
    textsize: float = 8,
    ylog: bool = True,
):
    if ax is None:
        _, ax = plt.subplots()
    md = np.abs(md)

    if colors is None:
        ax.scatter(bf, md, color="grey", s=s, alpha=0.8, linewidth=0)
    else:
        assert len(colors) == len(bf)
        color_mask_na = pd.isnull(colors)
        colors_uni = np.unique(colors[~color_mask_na]).tolist()
        if color_mask_na.any():
            colors_uni = [np.NaN] + colors_uni  # plot nan firstly!
        for colori in colors_uni:
            if pd.isnull(colori):
                maski = color_mask_na
                colori = "grey"
            else:
                maski = colors == colori
            ax.scatter(
                bf[maski], md[maski], color=colori, s=s * 2, linewidth=0
            )
    if x_thre is not None:
        ax.axvline(
            x=-x_thre, color="black", linestyle="--", linewidth=1.0, alpha=0.2
        )
        ax.axvline(
            x=x_thre, color="black", linestyle="--", linewidth=1.0, alpha=0.2
        )
    if y_thre is not None:
        ax.axhline(
            y=y_thre, color="black", linestyle="--", linewidth=1.0, alpha=0.2
        )

    ax.set_xlabel(r"$\log_e$(Bayes factor)", fontsize=fontsize)
    ax.set_ylabel("|Differential Metric|", fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    if ylog:
        ax.set_yscale("log")

    if anno is not None:
        idx = pd.notnull(anno).nonzero()[0]
        texts = []
        for i in idx:
            name = anno[i]
            x = bf[i]
            y = md[i]
            texts.append(
                ax.text(x=x, y=y, s=name, fontdict={"size": textsize})
            )
        adjust_text(
            texts,
            only_move={"texts": "xy"},
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="k", lw=0.5),
        )

    if title:
        ax.set_title(title, fontsize=fontsize)

    return ax

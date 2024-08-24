import warnings

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from scmidas.models import MIDAS
from scmidas.datasets import GenDataFromPath, GetDataInfo
import scmidas.utils as utils


warnings.filterwarnings("ignore")
sc.set_figure_params(figsize=(4, 4))


def run_midas(data_root: str, res_root):
    # These data has been pre-processed.
    data_path = [
        {
            "rna": f"{data_root}/processed/wnn_demo/subset_0/mat/rna.csv",
            "adt": f"{data_root}/processed/wnn_demo/subset_0/mat/adt.csv",
        },
        {"rna": f"{data_root}/processed/wnn_demo/subset_1/mat/rna.csv"},
        {"adt": f"{data_root}/processed/wnn_demo/subset_2/mat/adt.csv"},
    ]
    remove_old = False

    GenDataFromPath(
        data_path, f"{res_root}/data", remove_old
    )  # generate a directory, can be substituted by preprocess/split_mat.py
    data = [GetDataInfo(f"{res_root}/data")]
    model = MIDAS(data)
    model.init_model(model_path="./res/wnn/train/sp_latest.pt")
    # model.train(n_epoch=500, save_path=f"{res_root}/train")
    # model.viz_loss()
    # plt.savefig(f"{res_root}/loss.png")

    model.predict(joint_latent=True, save_dir=f"{res_root}/predict")
    emb = model.read_preds()
    c = emb["z"]["joint"][:, :32]  # biological information
    b = emb["z"]["joint"][:, 32:]  # batch information
    adata = sc.AnnData(c)
    adata2 = sc.AnnData(b)
    adata.obs["batch"] = emb["s"]["joint"].astype("str")
    adata2.obs["batch"] = emb["s"]["joint"].astype("str")
    label = pd.concat(
        [
            pd.read_csv(
                f"{data_root}/labels/p1_0/label_seurat/l1.csv", index_col=0
            ),
            pd.read_csv(
                f"{data_root}/labels/p2_0/label_seurat/l1.csv", index_col=0
            ),
            pd.read_csv(
                f"{data_root}/labels/p3_0/label_seurat/l1.csv", index_col=0
            ),
        ]
    )
    adata.obs["label"] = label.values.flatten()
    adata2.obs["label"] = label.values.flatten()

    sc.pp.subsample(adata, fraction=1)  # shuffle for better visualization
    sc.pp.subsample(adata2, fraction=1)

    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.pp.neighbors(adata2)
    sc.tl.umap(adata2)

    fig = sc.pl.umap(
        adata,
        color=["batch", "label"],
        wspace=0.2,
        title=["var c (batch)", "var c (celltype)"],
        return_fig=True,
    )
    fig.savefig(f"{res_root}/var_c_umap.png")
    fig = sc.pl.umap(
        adata2,
        color=["batch", "label"],
        wspace=0.2,
        title=["var u (batch)", "var u (celltype)"],
        return_fig=True,
    )
    fig.savefig(f"{res_root}/var_u_umap.png")

    model.predict(save_dir=f"{res_root}/predict", mod_latent=True)
    modality_emb = model.read_preds(joint_latent=True, mod_latent=True)
    label["s"] = modality_emb["s"]["joint"]
    label.columns = ["x", "s"]
    utils.viz_mod_latent(modality_emb, label, legend=False)
    plt.savefig(f"{res_root}/mod_latent.png")


def main():
    data_root = "./data/wnn"
    res_root = "./res/wnn"
    run_midas(data_root=data_root, res_root=res_root)


if __name__ == '__main__':
    main()

import os
import warnings
from typing import Sequence

import anndata as ad
import mudata as md
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scmidas.models import MIDAS
from scmidas.datasets import GenDataFromPath, GetDataInfo
import scmidas.utils as utils
from mmAAVI.preprocess import merge_obs_from_all_modalities, log1p_norm
from mmAAVI import MMAAVI
import scipy.sparse as sp
import biothings_client as bc


warnings.filterwarnings("ignore")
sc.set_figure_params(figsize=(4, 4))


def get_geneinfo(names: Sequence[str], return_df: bool = True) -> pd.DataFrame:
    cache_url = "./mygene_cache.sqlite"
    gene_client = bc.get_client("gene")
    gene_client.set_caching(cache_url)
    gene_meta = gene_client.querymany(
        names,
        species="human",
        scopes=["symbol", "alias"],
        fields="all",
        verbose=True,
        as_dataframe=return_df,
        df_index=True,
    )
    gene_client.stop_caching()
    return gene_meta


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
    # model.init_model(model_path=f"{res_root}/train/sp_latest.pt")
    model.init_model()
    model.train(n_epoch=500, save_path=f"{res_root}/train")
    model.viz_loss()
    plt.savefig(f"{res_root}/loss.png")

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


def run_mmAAVI(data_root: str, res_root: str):
    os.makedirs(res_root, exist_ok=True)

    data_path = {
        "rna": {
            "b0": f"{data_root}/processed/wnn_demo/subset_0/mat/rna.csv",
            "b1": f"{data_root}/processed/wnn_demo/subset_1/mat/rna.csv",
        },
        "adt": {
            "b0": f"{data_root}/processed/wnn_demo/subset_0/mat/adt.csv",
            "b2": f"{data_root}/processed/wnn_demo/subset_2/mat/adt.csv",
        },
    }
    label_path = {
        "b0": f"{data_root}/labels/p1_0/label_seurat/l1.csv",
        "b1": f"{data_root}/labels/p2_0/label_seurat/l1.csv",
        "b2": f"{data_root}/labels/p3_0/label_seurat/l1.csv",
    }

    adata_mods = {}
    for mod in ["rna", "adt"]:
        adata_mod = []
        for bk in ["b0", "b1", "b2"]:
            if bk not in data_path[mod]:
                continue
            df = pd.read_csv(data_path[mod][bk], index_col=0)
            adata = ad.AnnData(
                df.values,
                obs=pd.DataFrame(
                    pd.read_csv(label_path[bk], index_col=0).values,
                    index=df.index,
                    columns=["label"],
                ),
                var=pd.DataFrame(np.empty(df.shape[1]), index=df.columns),
            )
            adata.obs["batch"] = bk
            adata_mod.append(adata)
        adata_mod = ad.concat(adata_mod)
        adata_mods[mod] = adata_mod
    mdata = md.MuData(adata_mods)
    merge_obs_from_all_modalities(mdata, "label")
    merge_obs_from_all_modalities(mdata, "batch")

    # construct graph
    var_adt = mdata.mod["adt"].var.index.values
    var_rna = mdata.mod["rna"].var.index.values
    # prot_names = np.loadtxt("./data/proteins_alias.txt", dtype="U")
    # name_sets = []
    # for line in prot_names:
    #     name_sets.append(
    #         set(namei.lower() for namei in ([line[0]] + line[1].split(",")))
    #     )
    var_all = np.concatenate([var_adt, var_rna])
    geneinfos = get_geneinfo(var_all, return_df=True)
    name_sets = []
    for k in geneinfos.index.unique():
        subset = geneinfos.loc[[k], :]
        name_set = set(
            subset.index.to_list() + subset["symbol"].dropna().to_list()
        )
        for alias in subset["alias"]:
            if isinstance(alias, list):
                name_set.union(alias)
        name_sets.append(name_set)

    row, col = [], []
    for i, name_adt in enumerate(var_adt):
        for seti in name_sets:
            if name_adt in seti:
                break
        else:
            continue

        for j, name_rna in enumerate(var_rna):
            if name_rna in seti:
                row.append(j)
                col.append(i)
    row, col = np.array(row), np.array(col)
    rna_protein = sp.coo_array(
        (np.ones_like(row), (row, col)),
        shape=(var_rna.shape[0], var_adt.shape[0]),
    )
    net = sp.block_array(
        [
            [None, rna_protein],
            [rna_protein.T, None],
        ]
    )
    mdata.varp["net"] = sp.csr_matrix(net)  # anndata only support csr and csc

    # preprocess the count value
    for m in mdata.mod.keys():
        adatai = mdata.mod[m]
        adatai.obsm["log1p_norm"] = np.zeros(adatai.shape)
        for bi in adatai.obs["batch"].unique():
            maski = (adatai.obs["batch"] == bi).values
            datai = adatai.X[maski, :]
            # 1st pp method: log1p and normalization
            adatai.obsm["log1p_norm"][maski, :] = log1p_norm(datai)

    # run mmAAVI
    model = MMAAVI(
        dim_c=8,
        input_key="log1p_norm",
        net_key="net",
        balance_sample="max",
        num_workers=4,
    )
    model.fit(mdata)
    # mdata.obs["mmAAVI_c_label"] = mdata.obsm["mmAAVI_c"].argmax(axis=1)

    # plot figures
    sc.pp.neighbors(mdata, use_rep="mmAAVI_z")
    sc.tl.umap(mdata, min_dist=0.2)
    fig = sc.pl.umap(
        mdata,
        color=["batch", "label"],
        wspace=0.2,
        title=["z (batch)", "z (celltype)"],
        return_fig=True,
    )
    fig.savefig(f"{res_root}/z_umap.png")


def main():
    # run_midas(data_root="./data/wnn", res_root="./res/wnn/midas")
    run_mmAAVI(data_root="./data/wnn", res_root="./res/wnn/mmaavi")


if __name__ == "__main__":
    main()

import os
import os.path as osp
import re
from argparse import ArgumentParser

import numpy as np
import anndata as ad
import mudata as md
import scanpy as sc
from scipy import sparse as sp
from mmAAVI.preprocess import lsi
from mmAAVI.genomic import Bed
from mmAAVI.preprocess import merge_obs_from_all_modalities


def main():
    parser = ArgumentParser()
    parser.add_argument("--raw_data_dir", default="./data")
    parser.add_argument("--preproc_data_dir", default="./res")
    parser.add_argument("--preproc_data_name", default="triple")
    args = parser.parse_args()

    os.makedirs(args.preproc_data_dir, exist_ok=True)

    # ====== ATAC ======
    atac = ad.read_h5ad(osp.join(args.raw_data_dir, "10x-ATAC-Brain5k.h5ad"))
    sc.pp.highly_variable_genes(atac, flavor="cell_ranger")
    atac = atac[:, atac.var["highly_variable"]]
    atac.obs.index = [f"atac_{i}" for i in range(atac.shape[0])]
    atac.obs["batch"] = "b1-atac"
    atac.obs["cell_type_ori"] = atac.obs["cell_type"].copy()
    atac.obs["cell_type"] = atac.obs.cell_type.replace(
        {
            "L2/3 IT": "Layer2/3",
            "L4": "Layer5a",
            "L5 IT": "Layer5a",
            "L6 IT": "Layer5a",
            "L5 PT": "Layer5",
            "NP": "Layer5b",
            "L6 CT": "Layer6",
            "Vip": "CGE",
            "Pvalb": "MGE",
            "Sst": "MGE",
        }
    )

    # ====== RNA ======
    rna = ad.read_h5ad(osp.join(args.raw_data_dir, "Saunders-2018.h5ad"))
    rna = rna[:, rna.var["highly_variable"]]
    rna.obs.index = [f"rna_{i}" for i in range(rna.shape[0])]
    rna.obs["batch"] = "b2-rna"

    # ====== Methylation ======
    methy = ad.read_h5ad(osp.join(args.raw_data_dir, "Luo-2017.h5ad"))
    methy.layers["raw"] = methy.X.copy()
    methy.X = methy.layers["norm"].copy()
    methy.obs.index = [f"methy_{i}" for i in range(methy.shape[0])]
    methy.obs["batch"] = "b3-methy"
    methy.obs["cell_type_ori"] = methy.obs["cell_type"].copy()
    methy.obs["cell_type"] = methy.obs.cell_type.replace(
        {
            "mL2/3": "Layer2/3",
            "mL4": "Layer5a",
            "mL5-1": "Layer5a",
            "mDL-1": "Layer5a",
            "mDL-2": "Layer5a",
            "mL5-2": "Layer5",
            "mL6-1": "Layer5b",
            "mL6-2": "Layer6",
            "mDL-3": "Claustrum",
            "mVip": "CGE",
            "mNdnf-1": "CGE",
            "mNdnf-2": "CGE",
            "mPv": "MGE",
            "mSst-1": "MGE",
            "mSst-2": "MGE",
        }
    )

    # ====== graph (atac - rna) ======
    bed_rna = Bed(rna.var)
    bed_atac = Bed(atac.var)
    atac_rna = bed_atac.window_graph(
        bed_rna.expand(upstream=2e3, downstream=0), window_size=0
    )
    # ====== graph (rna - methylation) ======
    pattern = re.compile(r"(.*?)_mC[HG]")
    methy_var_genes = [pattern.search(namei).group(1) for namei in methy.var_names]
    row, col = [], []
    for i, methyi in enumerate(methy_var_genes):
        for j, rnaj in enumerate(rna.var_names):
            if methyi == rnaj:
                row.append(i)
                col.append(j)
    methy_rna = sp.coo_array(
        (np.ones(len(row)), (row, col)),
        shape=(len(methy_var_genes), rna.shape[1]),
    )
    # ====== filter feature by graph ======
    # atac: filter by highly_var and graph
    # rna: filter by highly_var
    mask_atac = atac_rna.sum(axis=1) > 0
    atac = atac[:, mask_atac]
    atac_rna = atac_rna.tocsr()[mask_atac, :]
    mask_met = methy_rna.sum(axis=1) > 0
    methy = methy[:, mask_met]
    methy_rna = methy_rna.tocsr()[mask_met, :]

    # ====== mudata ======
    net = sp.block_array(
        [
            [None, None, atac_rna],
            [None, None, methy_rna],
            [atac_rna.T, methy_rna.T, None],
        ]
    )
    mdata = md.MuData({"atac": atac, "met": methy, "rna": rna})
    mdata.varp["net"] = sp.csr_matrix(net)  # anndata only support csr and csc

    # ====== preprocess ======
    # only remain the features which are in the network
    # get the embedding for each data matrix
    n_embeds = 100
    for m in mdata.mod.keys():
        adatai = mdata.mod[m]
        adatai.obsm["lsi_pca"] = np.zeros((adatai.n_obs, n_embeds))
        for bi in adatai.obs["batch"].unique():
            maski = (adatai.obs["batch"] == bi).values
            datai = adatai.X[maski, :]
            # 2nd pp method: lsi for atac and pca for others
            if m == "atac":
                embedi = lsi(datai, n_components=n_embeds, n_iter=15)
            else:
                if sp.issparse(datai):
                    datai = sp.csr_matrix(datai)
                adata_cp = ad.AnnData(datai).copy()
                sc.pp.normalize_total(adata_cp)
                sc.pp.log1p(adata_cp)
                sc.pp.scale(adata_cp)
                sc.tl.pca(
                    adata_cp,
                    n_comps=100,
                    use_highly_variable=False,
                    svd_solver="auto",
                )
                embedi = adata_cp.obsm["X_pca"]
            adatai.obsm["lsi_pca"][maski, :] = embedi

    # ====== concat the labels ======
    merge_obs_from_all_modalities(mdata, "cell_type")
    merge_obs_from_all_modalities(mdata, "batch")

    # ====== save ======
    print(mdata)
    mdata.write(osp.join(args.preproc_data_dir, f"{args.preproc_data_name}.h5mu"))


if __name__ == "__main__":
    main()

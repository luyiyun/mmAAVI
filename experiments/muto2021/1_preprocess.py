import os
import os.path as osp
from argparse import ArgumentParser

import numpy as np
import anndata as ad
import mudata as md
import scanpy as sc
from scipy import sparse as sp
from mmAAVI.preprocess import lsi
from mmAAVI.genomic import Bed


def main():
    parser = ArgumentParser()
    parser.add_argument("--raw_data_dir", default="./data")
    parser.add_argument("--preproc_data_dir", default="./res")
    parser.add_argument("--preproc_data_name", default="muto2021")
    args = parser.parse_args()

    os.makedirs(args.preproc_data_dir, exist_ok=True)

    # ====== ATAC ======
    atac = ad.read_h5ad(osp.join(args.raw_data_dir, "Muto-2021-ATAC.h5ad"))
    sc.pp.highly_variable_genes(atac, flavor="cell_ranger")
    atac.obs.index = [f"atac_{i}" for i in range(atac.shape[0])]
    atac.obs["batch5"] = atac.obs["batch"]
    atac.obs["batch"] = atac.obs["batch"].map(lambda x: f"atac_{x}")

    # ====== RNA ======
    rna = ad.read_h5ad(osp.join(args.raw_data_dir, "Muto-2021-RNA.h5ad"))
    rna.obs.index = [f"rna_{i}" for i in range(rna.shape[0])]
    rna.obs["batch5"] = rna.obs["batch"]
    rna.obs["batch"] = rna.obs["batch"].map(lambda x: f"rna_{x}")

    # ====== graph ======
    bed_rna = Bed(rna.var)
    bed_atac = Bed(atac.var)
    atac_rna = bed_atac.window_graph(
        bed_rna.expand(upstream=2e3, downstream=0), window_size=0
    )
    net = sp.block_array(
        [
            [None, atac_rna],
            [atac_rna.T, None],
        ]
    )

    # ====== mudata ======
    mdata = md.MuData({"atac": atac, "rna": rna})
    mdata.varp["net"] = sp.csr_matrix(net)  # anndata only support csr and csc

    # ====== preprocess ======
    # only remain the features which are in the network
    mask = net.sum(axis=0) > 0
    mdata = mdata[:, mask]
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

    print(mdata)

    mdata.write(
        osp.join(args.preproc_data_dir, f"{args.preproc_data_name}.h5mu")
    )


if __name__ == "__main__":
    main()

import os
import os.path as osp
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import anndata as ad
import mudata as md
import biothings_client as bc
import scanpy as sc
from scipy import sparse as sp
from mmAAVI.genomic import Bed
from mmAAVI.preprocess import log1p_norm, lsi


def main():
    parser = ArgumentParser()
    parser.add_argument("--raw_data_dir", default="./data")
    parser.add_argument("--preproc_data_dir", default="./res")
    parser.add_argument("--preproc_data_name", default="mop5b")
    args = parser.parse_args()

    os.makedirs(args.preproc_data_dir, exist_ok=True)

    # =========================================================
    # load raw data and create MuData object
    # =========================================================
    batch_index = range(1, 6)
    # cell names
    obses = {}
    for i in batch_index:
        obsi = pd.read_csv(
            osp.join(args.raw_data_dir, f"meta_c{i}.csv"), index_col=0
        )
        obsi["batch"] = i
        obsi.index = obsi.index.map(lambda x: f"batch{i}:{x}")
        obsi.rename(columns={"cluster (remapped)": "cell_type"}, inplace=True)
        # obsi = obsi[
        #     ["cell_type", "cluster", "region", "MajorCluster", "SubCluster"]
        # ]
        obses[i] = obsi

    # atac
    fn = osp.join(args.raw_data_dir, "regions.txt")
    names = np.loadtxt(fn, dtype="U")
    atac_df = pd.Series(names).str.split("_", expand=True)
    assert not atac_df.isna().any().any()
    atac_df.columns = ["chrom", "chromStart", "chromEnd"]
    atac_df["chrom"] = atac_df["chrom"].str.slice(start=3)
    atac_df[["chromStart", "chromEnd"]] = atac_df[
        ["chromStart", "chromEnd"]
    ].astype(int)
    atac_df.index = names
    # counts
    atac = []
    for i in batch_index:
        fn = osp.join(args.raw_data_dir, f"RxC{i}.npz")
        if not osp.exists(fn):
            continue
        datai = sp.load_npz(fn).T
        datai = sp.csr_matrix(datai)
        adatai = ad.AnnData(datai, obs=obses[i], var=atac_df)
        atac.append(adatai)
    atac = ad.concat(
        atac, axis=0, merge="same"
    )  # "same" means only same var columns will be remained.

    # rna
    fn = osp.join(args.raw_data_dir, "genes.txt")
    rna_names = np.loadtxt(fn, dtype="U")
    # use biothings_client to get the location of genes in chromosome
    cache_url = "../mygene_cache.sqlite"
    gene_client = bc.get_client("gene")
    gene_client.set_caching(cache_url)
    gene_meta = gene_client.querymany(
        rna_names,
        species="human",
        scopes=["symbol"],
        fields=[
            "_score",
            "name",
            "genomic_pos.chr",
            "genomic_pos.start",
            "genomic_pos.end",
            "genomic_pos.strand",
        ],
        as_dataframe=True,
        df_index=True,
    )
    gene_client.stop_caching()
    gene_meta.rename(
        inplace=True,
        columns={
            "_score": "score",
            "genomic_pos.chr": "chrom",
            "genomic_pos.start": "chromStart",
            "genomic_pos.end": "chromEnd",
            "genomic_pos.strand": "strand",
        },
    )
    # only remain autosomal and sex chromosome genes
    gene_meta = gene_meta[
        gene_meta["chrom"].isin([str(i) for i in range(1, 23)] + ["X", "Y"])
    ]
    # remove duplicated genes
    mask_dup = gene_meta.index.duplicated(keep=False)
    assert not mask_dup.any()  # ensure there is no duplicated genes
    # reordering
    rna_df = gene_meta.reindex(index=rna_names)
    # postprocess the rna_df
    rna_df.fillna({"chrom": "."}, inplace=True)
    rna_df["strand"] = rna_df.strand.replace({1.0: "+", -1.0: "-"}).fillna("+")
    rna_df["chromStart"] = rna_df.chromStart.fillna(0.0)
    rna_df["chromEnd"] = rna_df.chromEnd.fillna(0.0)
    # must remain strand，it will be used when Bed.expand
    rna_df = rna_df[["chrom", "chromStart", "chromEnd", "strand"]]
    rna_df["strand"] = rna_df["strand"].astype(np.str_)
    rna_df.index = rna_df.index.astype(np.str_)
    # counts
    rna = []
    for i in batch_index:
        fn = osp.join(args.raw_data_dir, f"GxC{i}.npz")
        if not osp.exists(fn):
            continue
        datai = sp.load_npz(fn).T
        datai = sp.csr_matrix(datai)
        adatai = ad.AnnData(datai, obs=obses[i], var=rna_df)
        rna.append(adatai)
    rna = ad.concat(rna, axis=0, merge="same")

    # NOTE: the order of batches is 1, 3, 2, 4
    mdata = md.MuData({"atac": atac, "rna": rna})
    mdata.var_names_make_unique()

    # =========================================================
    # preprocessed the data
    # =========================================================
    # construct the network
    bed_rna = Bed(mdata.mod["rna"].var)
    bed_atac = Bed(mdata.mod["atac"].var)
    atac_rna = bed_atac.window_graph(
        bed_rna.expand(upstream=2e3, downstream=0),
        window_size=0,
        use_chrom=[str(i) for i in range(1, 23)] + ["X", "Y"],
    )
    net = sp.block_array([[None, atac_rna], [atac_rna.T, None]])
    mdata.varp["net"] = sp.csr_matrix(net)  # anndata only support csr and csc

    # filter features
    # only remain the features which are in the network
    mask = net.sum(axis=0) > 0
    mdata = mdata[:, mask]
    # get the embedding for each data matrix
    n_embeds = 100
    for m in mdata.mod.keys():
        adatai = mdata.mod[m]
        adatai.obsm["log1p_norm"] = np.zeros(adatai.shape)
        adatai.obsm["lsi_pca"] = np.zeros((adatai.n_obs, n_embeds))
        for bi in adatai.obs["batch"].unique():
            maski = (adatai.obs["batch"] == bi).values
            datai = adatai.X[maski, :]
            # 1st pp method: log1p and normalization
            adatai.obsm["log1p_norm"][maski, :] = log1p_norm(datai)
            # 2nd pp method: lsi for atac and pca for others
            if m == "atac":
                embedi = lsi(datai, n_components=100, n_iter=15)
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

    # =========================================================
    # save the preprocessed data
    # =========================================================
    print(mdata)
    mdata.write(
        osp.join(args.preproc_data_dir, f"{args.preproc_data_name}.h5mu")
    )


if __name__ == "__main__":
    main()

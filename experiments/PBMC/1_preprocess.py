import os
import os.path as osp
import logging
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import biothings_client as bc
from mudata import MuData
from scipy import sparse as sp
from tqdm import tqdm
from mmAAVI.genomic import Bed
from mmAAVI.preprocess import log1p_norm, lsi


def create_mosaic_dataset(root: str) -> MuData:
    logging.info("create mosaic dataset ......")
    # cell names
    obses = {}
    for i in range(1, 5):
        obsi = pd.read_csv(osp.join(root, f"meta_c{i}.csv"), index_col=0)
        obsi["batch"] = i
        obsi.index = obsi.index.map(lambda x: f"batch{i}:{x}")
        obses[i] = obsi

    # atac
    # peak names
    peak_names = np.loadtxt(osp.join(root, "regions.txt"), dtype="U")
    atac_df = pd.Series(peak_names).str.split("_", expand=True)
    assert not atac_df.isna().any().any()
    atac_df.columns = ["chrom", "chromStart", "chromEnd"]
    atac_df["chrom"] = atac_df["chrom"].str.slice(start=3)
    atac_df[["chromStart", "chromEnd"]] = atac_df[
        ["chromStart", "chromEnd"]
    ].astype(int)
    atac_df.index = peak_names.astype(np.str_)
    # counts
    atac = []
    for i in range(1, 5):
        fn = osp.join(root, f"RxC{i}.npz")
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
    rna_names = np.loadtxt(osp.join(root, "genes.txt"), dtype="U")
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
    gene_meta_nodup = gene_meta[~mask_dup]
    gene_meta_dup = gene_meta[mask_dup]
    remain_index = np.concatenate(
        [
            (
                (gene_meta_dup.index == "LINC00685")
                & (gene_meta_dup.chrom == "X")
            ).values.nonzero()[0],
            (
                (gene_meta_dup.index == "ZNF781")
                & (gene_meta_dup.chromStart == 37667751)
            ).values.nonzero()[0],
            (
                (gene_meta_dup.index == "ITFG2-AS1")
                & (gene_meta_dup.chromStart == 2695765)
            ).values.nonzero()[0],
            (
                (gene_meta_dup.index == "LINC02256")
                & (gene_meta_dup.chromStart == 32536047)
            ).values.nonzero()[0],
        ]
    )
    gene_meta = pd.concat(
        [gene_meta_nodup, gene_meta_dup.iloc[remain_index, :]]
    )
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
    for i in range(1, 5):
        fn = osp.join(root, f"GxC{i}.npz")
        if not osp.exists(fn):
            continue
        datai = sp.load_npz(fn).T
        datai = sp.csr_matrix(datai)
        adatai = ad.AnnData(datai, obs=obses[i], var=rna_df)
        rna.append(adatai)
    rna = ad.concat(rna, axis=0, merge="same")

    # protein
    fn = osp.join(root, "proteins_alias.txt")
    prot_names = np.loadtxt(fn, dtype="U")
    prot_ori = np.array([line[0] for line in prot_names], dtype=np.str_)
    prot_alias = np.array(
        [",".join(line) for line in prot_names], dtype=np.str_
    )
    prot_df = pd.DataFrame(dict(alias=prot_alias), index=prot_ori)
    # counts
    protein = []
    for i in range(1, 5):
        fn = osp.join(root, f"PxC{i}.npz")
        if not osp.exists(fn):
            continue
        datai = sp.load_npz(fn).T
        datai = datai.toarray()
        adatai = ad.AnnData(datai, obs=obses[i], var=prot_df)
        protein.append(adatai)
    protein = ad.concat(protein, axis=0, merge="same")

    # NOTE: the order of batches is 1, 3, 2, 4
    mdata = MuData({"atac": atac, "rna": rna, "protein": protein})
    mdata.var_names_make_unique()

    return mdata


def preprocess_mudata(mdata: MuData) -> MuData:
    logging.info("preprocess dataset ......")

    # === 1. construct graph ===
    # 1. atac-rna
    bed_rna = Bed(mdata.mod["rna"].var)
    bed_atac = Bed(mdata.mod["atac"].var)
    atac_rna = bed_atac.window_graph(
        bed_rna.expand(upstream=2e3, downstream=0),
        window_size=0,
        use_chrom=[str(i) for i in range(1, 23)] + ["X", "Y"],
    )
    # 2. rna-protein
    var_protein = mdata.mod["protein"].var
    var_rna = mdata.mod["rna"].var
    var_rna_index = var_rna.index.str.slice(4)
    row, col = [], []
    for i, p_alias in tqdm(
        enumerate(var_protein["alias"]), total=var_protein.shape[0]
    ):
        for j, g_symbol in enumerate(var_rna_index):
            if (g_symbol in p_alias) or (g_symbol.lower() in p_alias.lower()):
                row.append(j)
                col.append(i)
    row, col = np.array(row), np.array(col)
    rna_protein = sp.coo_array(
        (np.ones_like(row), (row, col)),
        shape=(var_rna.shape[0], var_protein.shape[0]),
    )
    # 3. global network
    # n_atac, n_rna, n_prot = (
    #     mdata.mod["atac"].n_vars,
    #     mdata.mod["rna"].n_vars,
    #     mdata.mod["protein"].n_vars,
    # )
    net = sp.block_array(
        [
            [None, atac_rna, None],
            [atac_rna.T, None, rna_protein],
            [None, rna_protein.T, None],
        ]
    )
    mdata.varp["net"] = sp.csr_matrix(net)  # anndata only support csr and csc

    # === 2. filter features ===
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

    return mdata


def main():
    parser = ArgumentParser()
    parser.add_argument("--raw_data_dir", default="./data")
    parser.add_argument("--preproc_data_dir", default="./res")
    parser.add_argument("--preproc_data_name", default="pbmc")
    args = parser.parse_args()

    os.makedirs(args.preproc_data_dir, exist_ok=True)
    mdata = create_mosaic_dataset(args.raw_data_dir)
    mdata = preprocess_mudata(mdata)
    # for subsize in subsample_sizes:
    #     for seedi in seeds:
    #         _, dat_used = mdat.split(subsize, seed=seedi)
    #         dat_used.save(
    #             osp.join(res_dir, "pbmc_%d_%d.mmod" % (dat_used.nobs, seedi))
    #         )

    logging.info("saving the pp data ......")
    mdata.write(
        osp.join(args.preproc_data_dir, f"{args.preproc_data_name}.h5mu")
    )


if __name__ == "__main__":
    main()

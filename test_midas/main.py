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
from mmAAVI.utils_dev import get_adata_from_mudata, plot_results_table
from mmAAVI.utils import setup_seed
from mmAAVI.genomic import Bed
from mmAAVI import MMAAVI
import scipy.sparse as sp
import biothings_client as bc
from scib_metrics.benchmark import Benchmarker, BioConservation


warnings.filterwarnings("ignore")
sc.set_figure_params(figsize=(4, 4))


class MIDAS_RUNNER:

    def __init__(self, res_root: str) -> None:
        self._res_root = res_root
        self._dim_c = 30
        self._latent_fn = f"{self._res_root}/latent.h5ad"
        self._latent_batch_fn = f"{self._res_root}/latent_batch.h5ad"

    def load_wnn_data(self, data_root: str):
        self._data_root = data_root

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

        self._label = pd.concat(
            [
                pd.read_csv(
                    f"{self._data_root}/labels/p1_0/label_seurat/l1.csv",
                    index_col=0,
                ),
                pd.read_csv(
                    f"{self._data_root}/labels/p2_0/label_seurat/l1.csv",
                    index_col=0,
                ),
                pd.read_csv(
                    f"{self._data_root}/labels/p3_0/label_seurat/l1.csv",
                    index_col=0,
                ),
            ]
        )

        # generate a directory, can be substituted by preprocess/split_mat.py
        GenDataFromPath(data_path, f"{self._res_root}/data", remove_old)
        self._data = [GetDataInfo(f"{self._res_root}/data")]

    def load_dogma_data(self, data_root: str):
        self._data_root = data_root

        data_path = [
            {"rna": f"{data_root}/processed/dogma_demo/subset_0/mat/rna.csv"},
            {
                "atac": (
                    f"{data_root}/processed/dogma_demo/subset_1/"
                    "mat/atac.csv"
                ),
                "rna": (
                    f"{data_root}/processed/dogma_demo/subset_1/" "mat/rna.csv"
                ),
            },
            {
                "rna": (
                    f"{data_root}/processed/dogma_demo/subset_2/" "mat/rna.csv"
                ),
                "adt": (
                    f"{data_root}/processed/dogma_demo/subset_2/" "mat/adt.csv"
                ),
            },
        ]
        self._label = pd.concat(
            [
                pd.read_csv(
                    f"{self._data_root}/labels/lll_ctrl/label_seurat/l1.csv",
                    index_col=0,
                ),
                pd.read_csv(
                    f"{self._data_root}/labels/lll_stim/label_seurat/l1.csv",
                    index_col=0,
                ),
                pd.read_csv(
                    f"{self._data_root}/labels/dig_ctrl/label_seurat/l1.csv",
                    index_col=0,
                ),
            ]
        )
        remove_old = False

        # generate a directory, can be substituted by preprocess/split_mat.py
        GenDataFromPath(data_path, f"{self._res_root}/data", remove_old)
        self._data = [GetDataInfo(f"{self._res_root}/data")]

    def run_model(self, trained_model: bool = False):
        self._trained_model = trained_model

        self._model = MIDAS(self._data)
        if trained_model:
            self._model.init_model(
                model_path=f"{self._res_root}/train/sp_latest.pt",
                dim_c=self._dim_c,
            )
        else:
            self._model.init_model(dim_c=self._dim_c)
            self._model.train(n_epoch=500, save_path=f"{self._res_root}/train")

        self._model.predict(
            joint_latent=True, save_dir=f"{self._res_root}/predict"
        )
        emb = self._model.read_preds()
        c = emb["z"]["joint"][:, : self._dim_c]  # biological information
        b = emb["z"]["joint"][:, self._dim_c :]  # batch information

        self._label.columns = ["label"]
        self._label["batch"] = emb["s"]["joint"].astype("str")
        self._label.index = self._label.apply(
            lambda ser: f"B{ser.loc['batch']}_{ser.name}", axis=1
        )
        self._adata = sc.AnnData(c, obs=self._label)
        self._adata2 = sc.AnnData(b, obs=self._label)
        self._adata.write_h5ad(self._latent_fn)
        self._adata2.write_h5ad(self._latent_batch_fn)

    def plot(self):
        if not self._trained_model:
            self._model.viz_loss()
            plt.savefig(f"{self._res_root}/loss.png")

        # shuffle for better visualization
        # sc.pp.subsample(adata, fraction=1)
        # sc.pp.subsample(adata2, fraction=1)

        sc.pp.neighbors(self._adata)
        sc.tl.umap(self._adata)
        sc.pp.neighbors(self._adata2)
        sc.tl.umap(self._adata2)

        fig = sc.pl.umap(
            self._adata,
            color=["batch", "label"],
            wspace=0.2,
            title=["var c (batch)", "var c (celltype)"],
            return_fig=True,
        )
        fig.savefig(f"{self._res_root}/var_c_umap.png")
        fig = sc.pl.umap(
            self._adata2,
            color=["batch", "label"],
            wspace=0.2,
            title=["var u (batch)", "var u (celltype)"],
            return_fig=True,
        )
        fig.savefig(f"{self._res_root}/var_u_umap.png")

        self._model.predict(
            save_dir=f"{self._res_root}/predict", mod_latent=True
        )
        modality_emb = self._model.read_preds(
            joint_latent=True, mod_latent=True
        )
        label = self._adata.obs["label"].to_frame(name="x")
        label["s"] = modality_emb["s"]["joint"]
        utils.viz_mod_latent(modality_emb, label, legend=False)
        plt.savefig(f"{self._res_root}/mod_latent.png")


class MMAAVI_RUNNER:

    @staticmethod
    def get_geneinfo(
        names: Sequence[str], return_df: bool = True
    ) -> pd.DataFrame:
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

    def __init__(self, res_root: str, dim_c: int = 8) -> None:
        self._res_root = res_root
        self._dim_c = dim_c
        self._latent_fn = f"{self._res_root}/latent.h5mu"

        os.makedirs(res_root, exist_ok=True)

    def load_wnn_data(self, data_root: str):
        self._data_root = data_root

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
        self._mdata = md.MuData(adata_mods)
        merge_obs_from_all_modalities(self._mdata, "label")
        merge_obs_from_all_modalities(self._mdata, "batch")

        # construct graph
        var_adt = self._mdata.mod["adt"].var.index.values
        var_rna = self._mdata.mod["rna"].var.index.values
        var_all = np.concatenate([var_adt, var_rna])
        geneinfos = self.get_geneinfo(var_all, return_df=True)
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
        # anndata only support csr and csc
        self._mdata.varp["net"] = sp.csr_matrix(net)

        # preprocess the count value
        for m in self._mdata.mod.keys():
            adatai = self._mdata.mod[m]
            adatai.obsm["log1p_norm"] = np.zeros(adatai.shape)
            for bi in adatai.obs["batch"].unique():
                maski = (adatai.obs["batch"] == bi).values
                datai = adatai.X[maski, :]
                # 1st pp method: log1p and normalization
                adatai.obsm["log1p_norm"][maski, :] = log1p_norm(datai)

    def load_dogma_data(self, data_root: str):
        self._data_root = data_root

        data_path = {
            "rna": {
                "b0": f"{data_root}/processed/dogma_demo/subset_0/mat/rna.csv",
                "b1": f"{data_root}/processed/dogma_demo/subset_1/mat/rna.csv",
                "b2": f"{data_root}/processed/dogma_demo/subset_2/mat/rna.csv",
            },
            "atac": {
                "b1": (
                    f"{data_root}/processed/dogma_demo/subset_1/"
                    "mat/atac.csv"
                ),
            },
            "adt": {
                "b2": f"{data_root}/processed/dogma_demo/subset_2/mat/adt.csv",
            },
        }
        label_path = {
            "b0": f"{self._data_root}/labels/lll_ctrl/label_seurat/l1.csv",
            "b1": f"{self._data_root}/labels/lll_stim/label_seurat/l1.csv",
            "b2": f"{self._data_root}/labels/dig_ctrl/label_seurat/l1.csv",
        }

        adata_mods = {}
        for mod in ["atac", "rna", "adt"]:
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
        self._mdata = md.MuData(adata_mods)
        merge_obs_from_all_modalities(self._mdata, "label")
        merge_obs_from_all_modalities(self._mdata, "batch")

        # construct graph - atac x rna
        var_atac = self._mdata.mod["atac"].var.index.values
        df_atac = pd.DataFrame.from_records(s.split("-") for s in var_atac)
        df_atac.columns = ["chrom", "chromStart", "chromEnd"]
        df_atac[["chromStart", "chromEnd"]].astype(int)
        df_atac["chrom"] = df_atac["chrom"].str.slice(3)

        var_rna = self._mdata.mod["rna"].var.index.values
        df_rna = self.get_geneinfo(var_rna, return_df=True)
        df_rna.rename(
            inplace=True,
            columns={
                "_score": "score",
                "genomic_pos.chr": "chrom",
                "genomic_pos.start": "chromStart",
                "genomic_pos.end": "chromEnd",
                "genomic_pos.strand": "strand",
            },
        )
        df_rna["chrom"] = df_rna["chrom"].map(
            lambda x: (
                x if x in ([str(i) for i in range(1, 23)] + ["X", "Y"])
                else "."
            )
        )  # only remain autosomal and sex chromosome genes
        # df_rna.fillna({"chrom": "."}, inplace=True)
        df_rna["strand"] = df_rna.strand.replace({1.0: "+", -1.0: "-"}).fillna(
            "+"
        )
        df_rna["chromStart"] = df_rna.chromStart.fillna(0.0)
        df_rna["chromEnd"] = df_rna.chromEnd.fillna(0.0)
        df_rna = df_rna[["chrom", "chromStart", "chromEnd", "strand"]]
        df_rna["strand"] = df_rna["strand"].astype(np.str_)
        # TODO: roughly remove duplicated items
        df_rna = df_rna[~df_rna.index.duplicated()]

        bed_rna = Bed(df_rna)
        bed_atac = Bed(df_atac)
        atac_rna = bed_atac.window_graph(
            bed_rna.expand(upstream=2e3, downstream=0),
            window_size=0,
            use_chrom=[str(i) for i in range(1, 23)] + ["X", "Y"],
        )

        # construct graph - rna x adt
        var_adt = self._mdata.mod["adt"].var.index.values
        var_rna = self._mdata.mod["rna"].var.index.values
        var_all = np.concatenate([var_adt, var_rna])
        geneinfos = self.get_geneinfo(var_all, return_df=True)
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
                [None, atac_rna, None],
                [atac_rna.T, None, rna_protein],
                [None, rna_protein.T, None],
            ]
        )
        # anndata only support csr and csc
        self._mdata.varp["net"] = sp.csr_matrix(net)

        # preprocess the count value
        for m in self._mdata.mod.keys():
            adatai = self._mdata.mod[m]
            adatai.obsm["log1p_norm"] = np.zeros(adatai.shape)
            for bi in adatai.obs["batch"].unique():
                maski = (adatai.obs["batch"] == bi).values
                datai = adatai.X[maski, :]
                # 1st pp method: log1p and normalization
                adatai.obsm["log1p_norm"][maski, :] = log1p_norm(datai)

    def run_model(self, trained_model: bool = False):
        if trained_model:
            self._mdata = md.read(f"{self._res_root}/latent.h5mu")
        else:
            model = MMAAVI(
                dim_c=self._dim_c,
                input_key="log1p_norm",
                net_key="net",
                balance_sample="max",
                num_workers=4,
            )
            model.fit(self._mdata)
            # mdata.obs["mmAAVI_c_label"] = \
            #   mdata.obsm["mmAAVI_c"].argmax(axis=1)
            self._mdata.write(self._latent_fn)

    def plot(self):
        sc.pp.neighbors(self._mdata, use_rep="mmAAVI_z")
        sc.tl.umap(self._mdata, min_dist=0.2)
        fig = sc.pl.umap(
            self._mdata,
            color=["batch", "label"],
            wspace=0.2,
            title=["z (batch)", "z (celltype)"],
            return_fig=True,
        )
        fig.savefig(f"{self._res_root}/z_umap.png")


def evaluate(
    midas_runner: MIDAS_RUNNER, mmaavi_runner: MMAAVI_RUNNER, res_root: str
):
    mdata = md.read(mmaavi_runner._latent_fn)
    res_midas = ad.read(midas_runner._latent_fn)

    mdata = get_adata_from_mudata(
        mdata, obs=["batch", "label"], obsm=["mmAAVI_z"]
    )
    mdata.obsm["MIDAS"] = res_midas[mdata.obs.index].X

    setup_seed(1234)
    bm = Benchmarker(
        mdata,
        batch_key="batch",
        label_key="label",
        embedding_obsm_keys=["mmAAVI_z", "MIDAS"],
        n_jobs=8,
        bio_conservation_metrics=BioConservation(
            nmi_ari_cluster_labels_kmeans=False,
            nmi_ari_cluster_labels_leiden=True,
        ),
    )
    bm.benchmark()

    res_df = bm.get_results(min_max_scale=False, clean_names=True)
    res_df = res_df.T
    res_df.rename(columns={"mmAAVI_z": "mmAAVI"}, inplace=True)
    embed_keys = ["mmAAVI", "MIDAS"]
    res_df[embed_keys] = res_df[embed_keys].astype(float)
    res_df.to_csv(f"{res_root}/eval_res.csv")

    metric_type = res_df["Metric Type"].values
    temp_df = res_df.drop(columns=["Metric Type"]).T
    temp_df["method"] = temp_df.index.map(lambda x: x.split("_")[0])
    plot_df = temp_df.groupby("method").mean().T
    plot_df["Metric Type"] = metric_type
    plot_results_table(plot_df, save_name=f"{res_root}/eval_table.png")


def main():
    # ========================================================================
    # wnn
    # ========================================================================
    # midas_runner = MIDAS_RUNNER(res_root="./res/wnn/midas")
    # midas_runner.load_wnn_data(data_root="./data/wnn/")
    # midas_runner.run_model()
    # midas_runner.plot()
    # mmaavi_runner = MMAAVI_RUNNER(res_root="./res/wnn/mmaavi")
    # mmaavi_runner.load_wnn_data(data_root="./data/wnn/")
    # mmaavi_runner.run_model()
    # mmaavi_runner.plot()
    # evaluate(midas_runner, mmaavi_runner, "./res/wnn")

    # ========================================================================
    # dogma
    # ========================================================================
    midas_runner = MIDAS_RUNNER(res_root="./res/dogma/midas")
    # midas_runner.load_dogma_data(data_root="./data/dogma/")
    # midas_runner.run_model(trained_model=True)
    # midas_runner.plot()
    mmaavi_runner = MMAAVI_RUNNER(res_root="./res/dogma/mmaavi")
    mmaavi_runner.load_dogma_data(data_root="./data/dogma/")
    mmaavi_runner.run_model()
    mmaavi_runner.plot()
    evaluate(midas_runner, mmaavi_runner, "./res/dogma")


if __name__ == "__main__":
    main()

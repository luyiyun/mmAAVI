import os
import os.path as osp
import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import mudata as md
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt

# import colorcet as cc
from tqdm import tqdm
import scanpy as sc


def main():
    parser = ArgumentParser()
    parser.add_argument("--results_dir", default="./res")
    parser.add_argument("--mmaavi_result", default="pbmc_decide_num_clusters")
    parser.add_argument("--compar_result", default="pbmc_comparison")
    parser.add_argument("--plotted_seed", default=0, type=int)
    parser.add_argument("--save_name", default="umap_pbmc")
    args = parser.parse_args()

    # ========================================================================
    # load trained results
    # ========================================================================
    os.makedirs(args.results_dir, exist_ok=True)
    mmaavi_result_fn = osp.join(args.results_dir, f"{args.mmaavi_result}.h5mu")
    mdata_mmaavi = md.read(mmaavi_result_fn)
    adata_compar = ad.read(
        osp.join(args.results_dir, f"{args.compar_result}.h5ad")
    )

    # ========================================================================
    # UMAP Visualization
    # ========================================================================
    label_name = "coarse_cluster"
    batch_name = "batch__code"
    nc_best = mdata_mmaavi.uns["best_nc"]
    embed_key = f"mmAAVI_nc{nc_best}_s{args.plotted_seed}_z"

    adata = ad.AnnData(
        obs=mdata_mmaavi.obs[[batch_name, label_name]],
        obsm={"mmAAVI": mdata_mmaavi.obsm[embed_key]},
    )
    for k in adata_compar.obsm.keys():
        if f"_s{args.plotted_seed}" in k:
            adata.obsm[k.split("_")[0]] = adata_compar.obsm[k]

    umap_keys = list(adata.obsm.keys())
    umap_arrs = {}
    for k in tqdm(umap_keys):
        sc.pp.neighbors(adata, use_rep=k, key_added=k, n_neighbors=30)
        sc.tl.umap(adata, neighbors_key=k, min_dist=0.2)
        umap_arrs[k] = adata.obsm["X_umap"]

    n_methods = len(umap_keys)
    batch_uni = np.unique(mdata_mmaavi.obs[batch_name].values)
    cell_type_uni = np.unique(mdata_mmaavi.obs[label_name].values)
    # fine_label_uni = np.unique(adata.obs[fine_label].values)
    markersize = 15000 / adata.n_obs
    markerscale = 10.0 / markersize

    batch_colors = sns.color_palette()
    label_colors = sns.color_palette()
    # colors_fine = sns.color_palette(cc.glasbey, n_colors=12)

    with warnings.catch_warnings(record=True):
        fig, axs = plt.subplots(
            ncols=2, nrows=n_methods, figsize=(3 * 4, n_methods * 3.5),
            squeeze=False
        )
        for i, key in tqdm(enumerate(umap_keys), total=n_methods):
            # sc.tl.umap(adata, neighbors_key=key, min_dist=0.2)
            xy = pd.DataFrame(umap_arrs[key], columns=["UMAP1", "UMAP2"])
            xy["batch"] = mdata_mmaavi.obs[batch_name].values
            xy["cell_type"] = mdata_mmaavi.obs[label_name].values
            # xy["fine_label"] = adata.obs[fine_label].values

            ax = axs[i, 0]
            for j, batch_i in enumerate(batch_uni):
                xyi = xy[xy["batch"] == batch_i]
                ax.plot(
                    xyi["UMAP1"],
                    xyi["UMAP2"],
                    ".",
                    markersize=markersize,
                    color=batch_colors[j],
                    label=batch_i,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel(key)

            if i == (n_methods - 1):
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles,
                    labels,
                    loc="upper center",
                    markerscale=markerscale,
                    frameon=False,
                    fancybox=False,
                    ncols=4,
                    bbox_to_anchor=(0.4, -0.2, 0.2, 0.2),
                    columnspacing=0.2,
                    handletextpad=0.1,
                )

            ax = axs[i, 1]
            for j, ct_i in enumerate(cell_type_uni):
                xyi = xy[xy["cell_type"] == ct_i]
                ax.plot(
                    xyi["UMAP1"],
                    xyi["UMAP2"],
                    ".",
                    markersize=markersize,
                    color=label_colors[j],
                    label=ct_i,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            if i == (n_methods - 1):
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles,
                    labels,
                    loc="upper center",
                    markerscale=markerscale,
                    frameon=False,
                    fancybox=False,
                    ncols=4,
                    bbox_to_anchor=(0.4, -0.2, 0.1, 0.2),
                    columnspacing=0.2,
                    handletextpad=0.1,
                )

            # ax = axs[i, 2]
            # for j, ct_i in enumerate(fine_label_uni):
            #     xyi = xy[xy["fine_label"] == ct_i]
            #     ax.plot(
            #         xyi["UMAP1"],
            #         xyi["UMAP2"],
            #         ".",
            #         markersize=markersize,
            #         color=colors_fine[j],
            #         label=ct_i,
            #     )
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.set_xlabel("")
            # ax.set_ylabel("")
            # if i == (n_methods - 1):
            #     handles, labels = ax.get_legend_handles_labels()
            #     ax.legend(
            #         handles,
            #         labels,
            #         loc="upper center",
            #         markerscale=markerscale,
            #         frameon=False,
            #         fancybox=False,
            #         ncols=4,
            #         bbox_to_anchor=(0.4, -0.2, 0.1, 0.2),
            #         columnspacing=0.2,
            #         handletextpad=0.1,
            #     )

        axs[0, 0].set_title("Batch")
        axs[0, 1].set_title("Cell Type")
        # axs[0, 2].set_title("Fine-grained Cell Type")

        # fig.tight_layout()
    # plt.show()

    fig.savefig(osp.join(args.results_dir, f"{args.save_name}.png"))


if __name__ == "__main__":
    main()

# sys.path.append(osp.abspath("../../src"))
# from mmAAVI.utils import read_json, setup_seed
# from mmAAVI.dataset import MosaicData

# # %%
# sys.path.append(osp.abspath("../"))
# from exper_utils import plot_results_table, save_figure, unintegrated_pca

# # %%
# root_mmAAVI = "./res/2_mmAAVI/"
# res_root = "./res/4_analysis/"
# res_root_compar = "./res/4_analysis/compar/"
# os.makedirs(res_root, exist_ok=True)
# os.makedirs(res_root_compar, exist_ok=True)

# # %%
# compar_methods = ["Unintegrated", "scmomat", "multimap", "stabmap", "uinmf"]
# compar_methods_name = {
#     "Unintegrated": "Unintegrated", "scmomat": "scMoMaT", "multimap": "MultiMap", "uinmf": "UINMF", "stabmap": "StabMap"
# }


# # %% [markdown]
# # # Load Dataset

# # %%
# dat = MosaicData.load("./res/1_pp/pbmc.mmod")
# label_name = "cell_type"
# fine_label = "fine_cell_type"
# print(dat)

# # %%
# adata = dat.to_anndata(sparse=True)
# adata.obsm["X_pca"] = unintegrated_pca(dat, K=30)
# adata.X = sp.csr_matrix(adata.X)
# adata.varp["window"] = sp.csr_matrix(adata.varp["window"])
# print(adata)

# # %% [markdown] jp-MarkdownHeadingCollapsed=true
# # # Evaluate benchmark methods

# # %% [markdown] jp-MarkdownHeadingCollapsed=true
# # ## Load embeddings

# # %%
# embeds = []
# for methodi in compar_methods:
#     if methodi == "Unintegrated":
#         embed = adata.obsm["X_pca"]
#         embeds.append((methodi, embed, 0))
#     else:
#         resi = osp.join("./res/3_comparison/", methodi)
#         for fn in os.listdir(resi):
#             match = re.search(r"pbmc_([0-9]).csv", fn)
#             if match:
#                 seedi = int(match.group(1))
#                 ffn = osp.join(resi, fn)
#                 embed = pd.read_csv(ffn, index_col=0).values
#                 embeds.append((methodi, embed, seedi))
# print(len(embeds))

# embed_keys = []
# for methodi, embed, seedi in embeds:
#     # 计算一下是否存在重复的embeddings
#     embed_df = pd.DataFrame(embed)
#     if embed_df.duplicated().any():
#         print("%s-%d has duplicates" % (methodi, seedi))
#         mask = embed_df.duplicated().values
#         embed[mask, :] = embed[mask, :] + np.random.randn(mask.sum(), embed.shape[1]) * 1e-3
#     key = "%s-%d" % (methodi, seedi)
#     adata.obsm[key] = embed
#     embed_keys.append(key)
# print(embed_keys)

# # %% [markdown] jp-MarkdownHeadingCollapsed=true
# # ## Calculate all scores for the benchmark methods use `scib`

# # %% [markdown]
# # ### Use coarse label

# # %% [markdown]
# # ### Use fine-grained label

# # %%
# bm = Benchmarker(
#     adata, batch_key="_batch", label_key=fine_label, embedding_obsm_keys=embed_keys, n_jobs=8,
#     bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_kmeans=False, nmi_ari_cluster_labels_leiden=True)
# )
# bm.benchmark()

# # %%
# res_df = bm.get_results(min_max_scale=False, clean_names=True)
# res_df = res_df.T
# res_df[embed_keys] = res_df[embed_keys].astype(float)

# res_df.to_csv(osp.join(res_root_compar, "benchmark_scib_fine.csv"))

# # %% [markdown]
# # # Evaluate the mmAAVI

# # %% [markdown] jp-MarkdownHeadingCollapsed=true
# # ## metrics scores of multiple runs

# # %%
# # 查找所有的运行结果
# runs = []
# for cdim in range(3, 11):
#     for di in os.listdir(root_mmAAVI):
#         if di.startswith("cdimension_%d_" % cdim):
#             runs.append((cdim, osp.join(root_mmAAVI, di)))
#             break
# runs

# # %%
# metric_scores = []
# for nc, model_dir_i in runs:
#     for seedi in os.listdir(model_dir_i):
#         model_dir_ii = osp.join(model_dir_i, seedi)
#         hist_valid = pd.read_csv(osp.join(model_dir_ii, "hist_valid.csv"), index_col=0)
#         best = read_json(osp.join(model_dir_ii, "best_score.csv"))
#         best_epoch = best["epoch"]
#         best_metric = hist_valid.loc[best_epoch, "metric"]
#         metric_scores.append({"nc": nc, "seed": seedi, "metric": best_metric})
# metric_scores = pd.DataFrame.from_records(metric_scores)
# # metric_scores.head()

# # %%
# # 计算mean+std
# nc_mean_std = metric_scores.groupby("nc")["metric"].apply(lambda x: np.mean(x)+np.std(x)).to_frame().reset_index()
# nc_mean_std.columns=["nc", "score"]
# # nc_mean_std.head()

# # %%
# color1, color2 = "tab:blue", "tab:red"
# fg = sns.relplot(data=metric_scores, x="nc", y="metric", kind="line", aspect=2, c=color1)
# fg.ax.set_xlabel("The number of mixture components")
# fg.ax.set_ylabel("The Validation Loss", color=color1)
# fg.ax.tick_params(axis='y', labelcolor=color1)

# ax2 = fg.ax.twinx()
# ax2.plot(nc_mean_std.nc, nc_mean_std.score, color=color2)
# ax2.set_ylabel("$\mu+\sigma$", color=color2)
# ax2.tick_params(axis='y', labelcolor=color2)

# fg.tight_layout()
# plt.show()

# # %%
# fg.savefig(osp.join(res_root, "metric_by_cdimensions.pdf"))
# fg.savefig(osp.join(res_root, "metric_by_cdimensions.png"), dpi=300)
# fg.savefig(osp.join(res_root, "metric_by_cdimensions.tiff"), dpi=300)

# # %% [markdown]
# # ## Load the embeddings generated by the mmAAVI

# # %%
# # model_dir, nc = './res/2_mmAAVI/cdimension_8_2023-08-23_19-43-36/', 8
# model_dir, nc = "./res/2_mmAAVI/cdimension8_/", 8

# # %%
# emb_keys = []
# for i in os.listdir(model_dir):
#     keyi = "mmAAVI-%s" % i
#     emb_keys.append(keyi)

#     model_dir_i = osp.join(model_dir, i)
#     embed = torch.load(osp.join(model_dir_i, "latents.pt"))
#     embed = embed.detach().cpu().numpy()
#     adata.obsm[keyi] = embed

# # %% [markdown]
# # ## Calculate all scores for the mmAAVI use `scib`

# # %% [markdown]
# # ### Use the coarse label

# # %%
# setup_seed(1234)
# bm = Benchmarker(
#     adata, batch_key="_batch", label_key=label_name, embedding_obsm_keys=emb_keys,
#     bio_conservation_metrics=BioConservation(
#         nmi_ari_cluster_labels_kmeans=False,
#         nmi_ari_cluster_labels_leiden=True
#     ), n_jobs=5
# )
# bm.benchmark()

# # %%
# res_df = bm.get_results(min_max_scale=False)
# res_df = res_df.T
# res_df[emb_keys] = res_df[emb_keys].astype(float)
# res_df.to_csv(osp.join(res_root, "benchmark_scib_mmAAVI_nc%d.csv" % nc))

# # %%
# res_df = pd.read_csv(osp.join(res_root, "benchmark_scib_mmAAVI_nc%d.csv" % nc), index_col=0)
# res_df_others = pd.read_csv(osp.join(res_root_compar, "benchmark_scib.csv"), index_col=0)
# res_df = pd.merge(res_df.drop(columns=["Metric Type"]), res_df_others, how="outer", left_index=True, right_index=True)
# metric_type = res_df["Metric Type"].values
# temp_df = res_df.drop(columns=["Metric Type"]).T
# temp_df["method"] = temp_df.index.map(lambda x: x.split("-")[0])
# plot_df = temp_df.groupby("method").mean().T
# plot_df["Metric Type"] = metric_type
# plot_df.rename(columns=compar_methods_name, inplace=True)
# plot_results_table(plot_df, save_name=osp.join(res_root, "scib_results_nc%d.svg" % nc))
# plt.show()

# # %% [markdown]
# # ### Use the fine-grained label

# # %%
# setup_seed(1234)
# bm = Benchmarker(
#     adata, batch_key="_batch", label_key=fine_label, embedding_obsm_keys=emb_keys,
#     bio_conservation_metrics=BioConservation(
#         nmi_ari_cluster_labels_kmeans=False,
#         nmi_ari_cluster_labels_leiden=True
#     ), n_jobs=5
# )
# bm.benchmark()

# # %%
# res_df = bm.get_results(min_max_scale=False)
# res_df = res_df.T
# res_df[emb_keys] = res_df[emb_keys].astype(float)
# res_df.to_csv(osp.join(res_root, "benchmark_scib_mmAAVI_nc%d_fine.csv" % nc))
# # res_df

# # %%
# res_df = pd.read_csv(osp.join(res_root, "benchmark_scib_mmAAVI_nc%d_fine.csv" % nc), index_col=0)
# res_df_others = pd.read_csv(osp.join(res_root_compar, "benchmark_scib_fine.csv"), index_col=0)
# res_df = pd.merge(res_df.drop(columns=["Metric Type"]), res_df_others, how="outer", left_index=True, right_index=True)
# metric_type = res_df["Metric Type"].values
# temp_df = res_df.drop(columns=["Metric Type"]).T
# temp_df["method"] = temp_df.index.map(lambda x: x.split("-")[0])
# plot_df = temp_df.groupby("method").mean().T
# plot_df["Metric Type"] = metric_type
# plot_df.rename(columns=compar_methods_name, inplace=True)
# plot_results_table(plot_df, save_name=osp.join(res_root, "scib_results_nc%d_fine.svg" % nc))
# plt.show()

# # %% [markdown]
# # # Show results for different sample sizes

# # %% [markdown]
# # TODO: need to reorganize the following codes...

# # %% [markdown]
# # ## mmAAVI

# # %%
# runs = []
# for di in os.listdir(root_mmAAVI):
#     if di.startswith("subample"):
#         _, ni, seedi, ti = di.split("_", 3)
#         ni, seedi = int(ni), int(seedi)
#         runs.append((
#             osp.join(root_mmAAVI, di, "0"),  # res_dir
#             ni,                              # nsamples
#             seedi,                           # random seed
#             osp.join("./res/1_pp/", "pbmc_graph_feats_%d_%d.mmod" % (ni, seedi))
#         ))
# len(runs)

# # %% [markdown]
# # ### coarse label

# # %%
# all_res_df = []
# for i, (fn, ni, seedi, dat_fn) in enumerate(runs):
#     print("%d/%d" % (i+1, len(runs)))
#     embed = torch.load(osp.join(fn, "latents.pt"))
#     embed = embed.detach().cpu().numpy()
#     subdat = MosaicData.load(dat_fn).to_anndata(sparse=True)
#     subdat.obsm["embed"] = embed

#     setup_seed(1234)
#     bm = Benchmarker(
#         subdat, batch_key="_batch", label_key=label_name, embedding_obsm_keys=["embed"],
#         bio_conservation_metrics=BioConservation(
#             nmi_ari_cluster_labels_kmeans=False,
#             nmi_ari_cluster_labels_leiden=True
#         ), n_jobs=5
#     )
#     bm.benchmark()

#     res_df = bm.get_results(min_max_scale=False)
#     res_df = res_df.T
#     res_df = res_df["embed"].astype(float).to_frame().T
#     res_df["nsample"] = ni
#     res_df["seed"] = seedi
#     res_df["method"] = "mmAAVI"

#     all_res_df.append(res_df)

# all_res_df = pd.concat(all_res_df, axis=0)
# df2 = pd.read_csv(osp.join(res_root, "benchmark_scib_mmAAVI_nc%d.csv" % nc), index_col=0)
# df2 = df2.drop(columns=["Metric Type"]).T
# df2["nsample"] = dat.nobs
# df2["seed"] = df2.index.map(lambda x: int(x.split("-")[-1]))
# df2["method"] = "mmAAVI"
# all_res_df = pd.concat([all_res_df, df2], axis=0)
# all_res_df.to_csv(osp.join(res_root, "benchmark_scib_mmAAVI_subsample.csv"))

# # %% [markdown]
# # ### fine label

# # %%
# all_res_df = []
# for i, (fn, ni, seedi, dat_fn) in enumerate(runs):
#     print("%d/%d" % (i+1, len(runs)))
#     embed = torch.load(osp.join(fn, "latents.pt"))
#     embed = embed.detach().cpu().numpy()
#     subdat = MosaicData.load(dat_fn).to_anndata(sparse=True)
#     subdat.obsm["embed"] = embed

#     setup_seed(1234)
#     bm = Benchmarker(
#         subdat, batch_key="_batch", label_key=fine_label, embedding_obsm_keys=["embed"],
#         bio_conservation_metrics=BioConservation(
#             nmi_ari_cluster_labels_kmeans=False,
#             nmi_ari_cluster_labels_leiden=True
#         ), n_jobs=5
#     )
#     bm.benchmark()

#     res_df = bm.get_results(min_max_scale=False)
#     res_df = res_df.T
#     res_df = res_df["embed"].astype(float).to_frame().T
#     res_df["nsample"] = ni
#     res_df["seed"] = seedi
#     res_df["method"] = "mmAAVI"

#     all_res_df.append(res_df)

# all_res_df = pd.concat(all_res_df, axis=0)
# # 加上之前得到的full dataset结果
# df2 = pd.read_csv(osp.join(res_root, "benchmark_scib_mmAAVI_nc%d_fine.csv" % nc), index_col=0)
# df2 = df2.drop(columns=["Metric Type"]).T
# df2["nsample"] = dat.nobs
# df2["seed"] = df2.index.map(lambda x: int(x.split("-")[-1]))
# df2["method"] = "mmAAVI"
# all_res_df = pd.concat([all_res_df, df2], axis=0)
# all_res_df.to_csv(osp.join(res_root, "benchmark_scib_mmAAVI_subsample_fine.csv"))

# # %% [markdown]
# # ## Benchmark methods

# # %%
# runs = []
# for methodi in ["uinmf", "scmomat", "multimap", "stabmap"]:
#     resi = osp.join("./res/3_comparison", methodi)
#     for fni in os.listdir(resi):
#         if re.search(r"pbmc_graph_feats_[0-9]*?_[0-9].csv", fni):
#             ni, seedi = fni[:-4].split("_")[-2:]
#             data_fn = osp.join("./res/1_pp/", fni[:-4] + ".mmod")
#             ni = int(ni)
#             seedi = int(seedi)
#             runs.append((methodi, osp.join(resi, fni), data_fn, ni, seedi))
# print(len(runs))
# # runs[:2]

# # %% [markdown]
# # ### coarse label

# # %%
# all_res_df = []
# for i, (methodi, res_fni, data_fni, ni, seedi) in enumerate(runs):
#     print("%d/%d" % (i+1, len(runs)))
#     embed = pd.read_csv(res_fni, index_col=0).values
#     subdat = MosaicData.load(data_fni).to_anndata(sparse=True)
#     subdat.obsm["embed"] = embed

#     setup_seed(1234)
#     bm = Benchmarker(
#         subdat, batch_key="_batch", label_key=label_name, embedding_obsm_keys=["embed"],
#         bio_conservation_metrics=BioConservation(
#             nmi_ari_cluster_labels_kmeans=False,
#             nmi_ari_cluster_labels_leiden=True
#         ), n_jobs=5
#     )
#     bm.benchmark()

#     res_df = bm.get_results(min_max_scale=False)
#     res_df = res_df.T
#     res_df = res_df["embed"].astype(float).to_frame().T
#     res_df["nsample"] = ni
#     res_df["seed"] = seedi
#     res_df["method"] = methodi

#     all_res_df.append(res_df)
# all_res_df = pd.concat(all_res_df, axis=0)

# # %%
# # 加上之前得到的full dataset结果
# df2 = pd.read_csv(osp.join(res_root_compar, "benchmark_scib.csv"), index_col=0)
# df2 = df2.drop(columns=["Metric Type"]).T
# df2["nsample"] = dat.nobs
# df2["seed"] = df2.index.map(lambda x: int(x.split("-")[-1]))
# df2["method"] = df2.index.map(lambda x: x.split("-")[0])
# all_res_df = pd.concat([all_res_df, df2], axis=0)
# all_res_df.to_csv(osp.join(res_root_compar, "benchmark_scib_subsample.csv"))

# # %% [markdown]
# # ### fine label

# # %%
# all_res_df = []
# for i, (methodi, res_fni, data_fni, ni, seedi) in enumerate(runs):
#     print("%d/%d" % (i+1, len(runs)))
#     embed = pd.read_csv(res_fni, index_col=0).values
#     subdat = MosaicData.load(data_fni).to_anndata(sparse=True)
#     subdat.obsm["embed"] = embed

#     setup_seed(1234)
#     bm = Benchmarker(
#         subdat, batch_key="_batch", label_key=fine_label, embedding_obsm_keys=["embed"],
#         bio_conservation_metrics=BioConservation(
#             nmi_ari_cluster_labels_kmeans=False,
#             nmi_ari_cluster_labels_leiden=True
#         ), n_jobs=5
#     )
#     bm.benchmark()

#     res_df = bm.get_results(min_max_scale=False)
#     res_df = res_df.T
#     res_df = res_df["embed"].astype(float).to_frame().T
#     res_df["nsample"] = ni
#     res_df["seed"] = seedi
#     res_df["method"] = methodi

#     all_res_df.append(res_df)
# all_res_df = pd.concat(all_res_df, axis=0)
# # all_res_df.head()

# # %%
# # 加上之前得到的full dataset结果
# df2 = pd.read_csv(osp.join(res_root_compar, "benchmark_scib_fine.csv"), index_col=0)
# df2 = df2.drop(columns=["Metric Type"]).T
# df2["nsample"] = dat.nobs
# df2["seed"] = df2.index.map(lambda x: int(x.split("-")[-1]))
# df2["method"] = df2.index.map(lambda x: x.split("-")[0])
# all_res_df = pd.concat([all_res_df, df2], axis=0)
# all_res_df.to_csv(osp.join(res_root_compar, "benchmark_scib_subsample_fine.csv"))

# # %% [markdown]
# # ## Visualization

# # %%
# res_subsample = pd.concat([
#     pd.read_csv(osp.join(res_root, "benchmark_scib_mmAAVI_subsample.csv"), index_col=0),
#     pd.read_csv(osp.join(res_root_compar, "benchmark_scib_subsample.csv"), index_col=0)
# ], axis=0)
# res_subsample.replace({"method": compar_methods_name}, inplace=True)

# # %%
# # 绘制图形
# fg = sns.catplot(data=res_subsample, x="nsample", y="Total", hue="method", kind="box", aspect=2)
# sns.stripplot(data=res_subsample, x="nsample", y="Total", hue="method", ax=fg.ax, dodge=True, legend=False, alpha=1, s=5)
# fg.set_xlabels("")
# fg.set_ylabels("")
# fg.savefig(osp.join(res_root, "scib_subampling_total.pdf"))
# fg.savefig(osp.join(res_root, "scib_subampling_total.png"), dpi=300)
# fg.savefig(osp.join(res_root, "scib_subampling_total.tiff"), dpi=300)
# # fg = sns.relplot(data=res_subsample, x="nsample", y="Total", hue="method", kind="line", aspect=2)
# plt.show()

# # %%
# res_subsample = pd.concat([
#     pd.read_csv(osp.join(res_root, "benchmark_scib_mmAAVI_subsample_fine.csv"), index_col=0),
#     pd.read_csv(osp.join(res_root_compar, "benchmark_scib_subsample_fine.csv"), index_col=0)
# ], axis=0)
# res_subsample.replace({"method": compar_methods_name}, inplace=True)

# # %%
# # 绘制图形
# fg = sns.catplot(data=res_subsample, x="nsample", y="Total", hue="method", kind="box", aspect=2)
# sns.stripplot(data=res_subsample, x="nsample", y="Total", hue="method", ax=fg.ax, dodge=True, legend=False, alpha=1, s=5)
# fg.set_xlabels("")
# fg.set_ylabels("")
# fg.legend.set_title("")
# fg.savefig(osp.join(res_root, "scib_subampling_total_fine.pdf"))
# fg.savefig(osp.join(res_root, "scib_subampling_total_fine.png"), dpi=300)
# fg.savefig(osp.join(res_root, "scib_subampling_total_fine.tiff"), dpi=300)
# # fg = sns.relplot(data=res_subsample, x="nsample", y="Total", hue="method", kind="line", aspect=2)
# plt.show()

# # %%

import os
import os.path as osp
from argparse import ArgumentParser

import scanpy as sc
import mudata as md
import anndata as ad
from mmAAVI import MMAAVI
from mmAAVI.preprocess import merge_obs_from_all_modalities
from scib import metrics as scme


def main():
    parser = ArgumentParser()
    parser.add_argument("--preproc_data_dir", default="./res")
    parser.add_argument("--preproc_data_name", default="muto2021")
    parser.add_argument("--results_dir", default="./res")
    parser.add_argument("--results_name", default="muto2021_fit_once")
    args = parser.parse_args()

    batch_name, label_name = "batch", "cell_type"
    os.makedirs(args.results_dir, exist_ok=True)
    save_dir = args.results_dir
    save_name = args.results_name

    mdata = md.read(osp.join(
        args.preproc_data_dir, f"{args.preproc_data_name}.h5mu"
    ))
    merge_obs_from_all_modalities(mdata, key=batch_name)
    merge_obs_from_all_modalities(mdata, key=label_name)

    model = MMAAVI(
        dim_c=13,
        input_key="lsi_pca",
        net_key="net",
        balance_sample="max",
        num_workers=8,
        batch_key=batch_name,
        batch_size=256,
        disc_gradient_weight=50,
    )
    model.fit(mdata)
    mdata.obs["mmAAVI_c_label"] = mdata.obsm["mmAAVI_c"].argmax(axis=1)

    sc.pp.neighbors(mdata, use_rep="mmAAVI_z")
    sc.tl.leiden(mdata, resolution=0.1, key_added="leiden")

    model.differential(mdata, "leiden")

    sc.tl.umap(mdata, min_dist=0.2)
    # convert categorical
    mdata.obs[batch_name] = mdata.obs[batch_name].astype("category")
    mdata.obs["mmAAVI_c_label"] = mdata.obs["mmAAVI_c_label"].astype(
        "category"
    )
    # plot and save umap
    fig_umap = sc.pl.umap(
        mdata,
        color=[batch_name, label_name, "mmAAVI_c_label", "leiden"],
        ncols=2,
        return_fig=True,
    )
    fig_umap.savefig(osp.join(save_dir, f"{save_name}.png"))
    mdata.write(osp.join(save_dir, f"{save_name}.h5mu"))

    adata = ad.AnnData(obs=mdata.obs[["leiden", label_name]])
    ari = scme.ari(adata, cluster_key="leiden", label_key=label_name)
    nmi = scme.nmi(adata, cluster_key="leiden", label_key=label_name)
    print(f"ARI = {ari:.4f}, NMI = {nmi:.4f}")


if __name__ == "__main__":
    main()


# # %%
# # running for different dimension of c
# data_fn = osp.join(data_dir, "muto2021.mmod")
# for nc in [10, 11, 12] + [14, 15, 16, 17]:
#     print("nc = %d" % nc)
#     cfg_cp = deepcopy(cfg)
#     cfg_cp.loader["batch_size"] = 256
#     cfg_cp.model_common["nclusters"] = nc
#     cfg_cp.weights.update(
#         {
#             "weight_dot": 0.99,
#             "weight_mlp": 0.01,
#             "alpha": 50,
#         }
#     )
#     cfg_cp.train.update(
#         {
#             "early_stop_patient": 20,
#         }
#     )
#     run(
#         cfg=cfg_cp,
#         dat=data_fn,
#         save_root=res_dir,
#         trial_prefix="repeated_trial_%d_" % nc,
#         seeds=range(6),
#     )

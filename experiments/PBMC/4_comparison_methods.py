import os
import os.path as osp
from time import perf_counter
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import anndata as ad
import mudata as md
import scanpy as sc
import scmomat
import MultiMAP
from mmAAVI.preprocess import merge_obs_from_all_modalities


# for subsampling experiments
# data_fns = [
#     osp.join(data_dir, fn)
#     for fn in os.listdir(data_dir)
#     if re.search(r"pbmc_[0-9]*?_[0-9].mmod", fn)
# ]
# for i, data_fni in enumerate(data_fns):
#     res_fn = osp.join(res_dir, "%s.csv" % osp.basename(data_fni)[:-5])
#     print("%d/%d %s" % (i+1, len(data_fns), res_fn))
#     run(
#         data_fn=data_fni,
#         res_fn=res_fn,
#         use_pseduo=True,
#         K=30,
#         seed=0,
#         device="cuda:0",
#     )


def main():
    parser = ArgumentParser()
    parser.add_argument("--preproc_data_dir", default="./res")
    parser.add_argument("--preproc_data_name", default="pbmc")
    parser.add_argument("--results_dir", default="./res")
    parser.add_argument("--results_name", default="pbmc_comparison")
    parser.add_argument("--not_use_pseudo", action="store_true")
    parser.add_argument("--seeds", default=list(range(6)), type=int, nargs="+")
    parser.add_argument("--scmomat_device", default="cuda:0")
    parser.add_argument(
        "--methods", default=("scmomat", "multimap", "uinmf"), nargs="+"
    )
    args = parser.parse_args()

    # ========================================================================
    # load preprocessed data
    # ========================================================================
    print("-- load prprocessed data --")
    mdata_fn = osp.join(
        args.preproc_data_dir, f"{args.preproc_data_name}.h5mu"
    )
    os.makedirs(args.results_dir, exist_ok=True)
    mdata = md.read(mdata_fn)
    merge_obs_from_all_modalities(mdata, key="coarse_cluster")
    merge_obs_from_all_modalities(mdata, key="batch")
    print(mdata)

    # prepare the container to hold the results
    res_adata = ad.AnnData(obs={"placeholder": np.arange(mdata.n_obs)})

    # ========================================================================
    # rearrange the data
    # ========================================================================
    print("-- rearrange data --")
    batch_name = "batch"

    batch_uni = mdata.obs[batch_name].unique()
    batch_uni.sort()
    nbatches = batch_uni.shape[0]

    counts = {}
    for k, adat in mdata.mod.items():
        batch_uni_k = adat.obs[batch_name].unique()
        counts_k = []
        for bi in batch_uni:
            if bi in batch_uni_k:
                counts_k.append(adat.X[adat.obs[batch_name] == bi, :])
            else:
                counts_k.append(None)
        counts[k] = counts_k

    if not args.not_use_pseudo:
        net = mdata.varp["net"]
        atac_rna = net[mdata.varm["atac"], :][:, mdata.varm["rna"]].toarray()

        for i, arr_rna in enumerate(counts["rna"]):
            if arr_rna is None:
                arr_atac = counts["atac"][i]
                if arr_atac is not None:
                    counts["rna"][i] = ((arr_atac @ atac_rna) != 0).astype(int)

    # ========================================================================
    # running scmomat
    # ========================================================================
    if "scmomat" in args.methods:
        print("-- scmomat: preprocessing --")
        counts_scmomat = defaultdict(list)
        for k, arrs in counts.items():
            for dati in arrs:
                if dati is None:
                    counts_scmomat[k].append(None)
                    continue

                if sp.issparse(dati):
                    dati = dati.toarray()
                if k == "atac":
                    dati = scmomat.preprocess(dati, modality="ATAC")
                else:
                    dati = scmomat.preprocess(
                        dati, modality="RNA", log=(k == "protein")
                    )
                counts_scmomat[k].append(dati)

        counts_scmomat["nbatches"] = nbatches
        counts_scmomat["feats_name"] = {
            k: adati.var.index.values for k, adati in mdata.mod.items()
        }

        print("-- scmomat: modeling --")
        res_timing = []
        for seedi in args.seeds:
            print(f"-- scmomat: modeling, seed is {seedi} --")
            start_time = perf_counter()
            model = scmomat.scmomat_model(
                counts=counts_scmomat,
                K=30,
                batch_size=0.1,
                interval=1000,
                lr=1e-2,
                lamb=0.001,
                seed=seedi,
                device=args.scmomat_device,
            )
            model.train_func(T=4000)
            end_time = perf_counter()
            res_timing.append((seedi, end_time - start_time))

            zs = model.extract_cell_factors()
            res_adata.obsm[f"scMoMaT_s{seedi}"] = np.concatenate(zs)

        res_adata.uns["timing"] = {"scMoMaT": res_timing}

    # ========================================================================
    # running multimap
    # ========================================================================
    if "multimap" in args.methods:
        print("-- multimap: preprocessing --")
        adatas = []
        for i, bi in enumerate(batch_uni):
            adatas_i = []
            for k in ["atac", "rna", "protein"]:
                dati = counts[k][i]
                if dati is not None:
                    var = mdata.mod[k].var.copy()
                    # duplicated varnames will cause error
                    var.index = [
                        f"{vname}_{i}" for i, vname in enumerate(var.index)
                    ]
                    adatai = ad.AnnData(
                        X=dati,
                        var=var,
                        obs=mdata.obs[mdata.obs[batch_name] == bi],
                    )
                    adatas_i.append(adatai)
            adatas.append(ad.concat(adatas_i, axis=1))

        for adatas_i in adatas:
            sc.pp.log1p(adatas_i)
            sc.pp.scale(adatas_i)
            sc.pp.pca(adatas_i)

        print("-- multimap: modeling --")
        res_timing = []
        for seedi in args.seeds:
            print(f"-- multimap: modeling, seed is {seedi} --")
            start_time = perf_counter()
            multimap_res = MultiMAP.Integration(
                adatas=adatas,
                use_reps=["X_pca"] * len(adatas),
                n_components=30,
                seed=seedi,
            )
            end_time = perf_counter()
            res_timing.append((seedi, end_time - start_time))

            res_adata.obsm[f"MultiMap_s{seedi}"] = multimap_res.obsm[
                "X_multimap"
            ]

        res_adata.uns["timing"] = {"MultiMap": res_timing}

    # ========================================================================
    # running multimap
    # ========================================================================
    if "uinmf" in args.methods:
        pass

    # ========================================================================
    # save the results
    # ========================================================================
    print("-- save the results --")
    res_adata.write(osp.join(args.results_dir, f"{args.results_name}.h5ad"))


if __name__ == "__main__":
    main()

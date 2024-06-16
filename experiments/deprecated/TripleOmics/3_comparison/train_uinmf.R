# 1. 设置环境
renv::activate(project = normalizePath("../../../environments/env_uinmf/"))

# 2. 载入包和python函数
library(rliger)
library(Seurat)
library(stringr)
library(reticulate)

use_condaenv("mosaic", conda = "~/mambaforge/bin/conda")
mmAAVI <- import_from_path("mmAAVI", normalizePath("../../../src/"))


pp_one_omics <- function(count_lists, use_seurat_select, num_select_genes = 2000, ...) {
  liger <- createLiger(count_lists, remove.missing = FALSE)
  liger <- normalize(liger, remove.missing = FALSE)

  if (use_seurat_select) {
    norms <- liger@norm.data
    norms <- purrr::reduce(norms, cbind)
    se <- CreateSeuratObject(norms)
    vars_2000 <- FindVariableFeatures(
      se, selection.method = "vst",
      nfeatures = num_select_genes
    )
    top2000 <- head(VariableFeatures(vars_2000), num_select_genes)
    # top2000_feats <- norms[top2000,]
    # liger <- selectGenes(liger)
    liger@var.genes <- top2000
  } else {
    liger <- selectGenes(liger, ...)
  }
  liger <- scaleNotCenter(liger)

  return(liger)
}


run <- function(data_fn, res_fn, use_pseduo = TRUE, K = 30, seed = 0) {
  set.seed(seed)
  dat <- mmAAVI$dataset$MosaicData$load(data_fn)
  dat$sparse_array2matrix()  # 使用py将所有的sparse改为matrix格式，r可以直接识别

  # net
  if (use_pseduo) {
    net1 <- dat$nets["window"]["atac", "rna"]
    net2 <- abs(dat$nets["window"]["met", "rna"])
  }
  # X
  grid <- list(
    atac = list(),
    met = list(),
    rna = list()
  )
  for (omic_name in c("atac", "rna", "met")) {
    for (bi in c("batch2", "batch3", "batch1")) {  # 注意必须保持这个顺序
      dati <- dat$X[bi, omic_name]

      if (!is.null(dati)) {
        rownames(dati) <- paste0(bi, "#", 1:nrow(dati))  # batch_names里面有-
        colnames(dati) <- paste0(omic_name, "-", 1:ncol(dati))
        grid[[omic_name]][[bi]] <- t(dati)
      } else if (use_pseduo & omic_name == "rna") {
        atac_i <- dat$X[bi, "atac"]
        met_i <- dat$X[bi, "met"]
        if (!is.null(atac_i)) {
          rna_i <- atac_i %*% net1
        } else if (!is.null(met_i)) {
          rna_i <- met_i %*% net2
        }
        rownames(rna_i) <- paste0(bi, "#", 1:nrow(rna_i))
        colnames(rna_i) <- paste0(omic_name, "-", 1:ncol(rna_i))
        grid[[omic_name]][[bi]] <- t(rna_i)
      }
    }
  }

  atac <- pp_one_omics(grid$atac, TRUE, 2000)
  met <- pp_one_omics(grid$met, TRUE, 2000)
  rna <- pp_one_omics(grid$rna, FALSE)

  for (namei in names(atac@scale.data)) {
    rna@scale.unshared.data[[namei]] <- t(atac@scale.data[[namei]])
    rna@var.unshared.features[[namei]] <- colnames(atac@scale.data[[namei]])
  }
  for (namei in names(met@scale.data)) {
    rna@scale.unshared.data[[namei]] <- t(met@scale.data[[namei]])
    rna@var.unshared.features[[namei]] <- colnames(met@scale.data[[namei]])
  }

  res_liger <- optimizeALS(
    rna, k=K, use.unshared = TRUE,
    max.iters = 30, thresh = 1e-10, rand.seed = seed,
    # remove.missing = FALSE
  )
  res_liger <- quantile_norm(res_liger)
  H <- as.data.frame.matrix(res_liger@H.norm)

  # 把顺序重新调换过来
  blabel <- sapply(strsplit(row.names(H), "#"), function(x) x[1])
  Hs <- list()
  for (bi in dat$batch_names) {
    mask <- blabel == bi
    Hs[[bi]] <- H[mask,]
  }
  H <- Reduce(rbind, Hs)

  write.csv(H, file = res_fn)
}


# 3. 读入数据
data_dir <- normalizePath("../res/1_pp/")
res_dir <- normalizePath("../res/3_comparison/uinmf/")
if (!dir.exists(res_dir)) { dir.create(res_dir) }

for (seedi in 0:5) {
  data_fn <- file.path(data_dir, "TripleOmics.mmod")
  res_fn <- file.path(res_dir, paste0("TripleOmics_all_", seedi, ".csv"))
  print(res_fn)
  run(data_fn, res_fn, use_pseduo = TRUE, K = 30, seed = seedi)
}

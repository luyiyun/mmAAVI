# 1. 设置环境
renv::activate(project = normalizePath("../../../environments/env_uinmf/"))

# 2. 载入包和python函数
library(rliger)
library(Seurat)
library(stringr)
library(reticulate)

use_condaenv("mosaic", conda = "~/mambaforge/bin/conda")
mmAAVI <- import_from_path("mmAAVI", normalizePath("../../../src/"))

pp_one_omics <- function(count_lists, use_seurat_select, num_select_genes = 2000) {
  liger <- createLiger(count_lists, remove.missing = FALSE)
  liger <- normalize(liger)

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
    liger <- selectGenes(liger, num.genes = num_select_genes)
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
    net <- dat$nets["window"]["atac", "rna"]
  }
  # X
  grid <- list()
  for (omic_name in c("atac", "rna", "protein")) {
    grid_i <- list()
    for (bi in dat$batch_names) {
      dati <- dat$X[bi, omic_name]
      if (!is.null(dati)) {
        rownames(dati) <- paste0(bi, "-", 1:nrow(dati))
        colnames(dati) <- paste0(omic_name, "-", 1:ncol(dati))
        grid_i[[bi]] <- t(dati)
      }
    }
    grid[[omic_name]] <- grid_i
  }

  atac <- pp_one_omics(grid$atac, TRUE, 2000)
  rna <- pp_one_omics(grid$rna, TRUE, 2000)
  protein <- pp_one_omics(grid$protein, FALSE, NULL)

  # 将顺序重新排列一下!
  for (namei in names(rna@scale.data)) {
    protein@scale.unshared.data[[namei]] <- t(rna@scale.data[[namei]])
    protein@var.unshared.features[[namei]] <- colnames(rna@scale.data[[namei]])
  }
  for (namei in names(atac@scale.data)) {
    protein@scale.unshared.data[[namei]] <- t(atac@scale.data[[namei]])
    protein@var.unshared.features[[namei]] <- colnames(atac@scale.data[[namei]])
  }

  res_liger <- optimizeALS(
    protein, k=K, use.unshared = TRUE,
    max.iters = 30, thresh = 1e-10, rand.seed = seed
  )
  res_liger <- quantile_norm(res_liger)
  H <- as.data.frame.matrix(res_liger@H.norm)
  write.csv(H, file = res_fn)
}

# 3. 读入数据
data_dir <- normalizePath("../res/1_pp/")
res_dir <- normalizePath("../res/3_comparison/uinmf/")
if (!dir.exists(res_dir)) { dir.create(res_dir) }

for (seedi in 0:5) {
  data_fn <- file.path(data_dir, "pbmc.mmod")
  res_fn <- file.path(res_dir, paste0("pbmc_", seedi, ".csv"))
  print(res_fn)
  run(data_fn, res_fn, use_pseduo = TRUE, K = 30, seed = seedi)
}

# data_fns <- c()
# for (data_fn in list.files(data_dir)) {
#   if (grepl("pbmc_[0-9]*?_[0-9].mmod", data_fn)) {
#     data_fns <- c(data_fns, data_fn)
#   }
# }
# for (i in seq_along(data_fns)) {
#   data_fn <- data_fns[i]
#   data_fn_full <- file.path(data_dir, data_fn)
#   res_fn <- file.path(res_dir, paste0(substr(data_fn, 0, nchar(data_fn)-4), "csv"))
#   print(paste0(i, "/", length(data_fns), ", ", res_fn))
#
#   run(data_fn_full, res_fn, use_pseduo = TRUE, K = 30, seed = 0)
# }

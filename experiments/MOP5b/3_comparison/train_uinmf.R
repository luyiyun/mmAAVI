# 1. 设置环境
setwd("~/Documents/mosaic-GAN/experiments/MOP5b/3_comparison")
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
    net <- dat$nets["window"]["atac", "rna"]
  }
  # X
  grid <- list()
  for (omic_name in c("atac", "rna")) {
    grid_i <- list()
    # 让atac存在的batches都在前面，因为atac要作为unshared部分，
    # 查看源码https://github.com/welch-lab/liger/blob/master/R/rliger.R:6866-6876
    # 可以知道，一旦unshared部分是“A-空-B”的形式，则B会被提前到“空”的位置来匹配
    # shared部分，这是不对的。
    for (bi in c("batch1", "batch3", "batch5", "batch2", "batch4")) {
      dati <- dat$X[bi, omic_name]
      
      if (!is.null(dati)) {
        rownames(dati) <- paste0(bi, "-", 1:nrow(dati))
        colnames(dati) <- paste0(omic_name, "-", 1:ncol(dati))
        grid_i[[bi]] <- t(dati)
      } else if (use_pseduo & omic_name == "rna") {
        atac_i <- dat$X[bi, "atac"]
        if (!is.null(atac_i)) {
          rna_i <- atac_i %*% net
          rownames(rna_i) <- paste0(bi, "-", 1:nrow(rna_i))
          colnames(rna_i) <- paste0(omic_name, "-", 1:ncol(rna_i))
          grid_i[[bi]] <- t(rna_i)
        }
      }
    }
    grid[[omic_name]] <- grid_i
  }
  
  atac <- pp_one_omics(grid$atac, TRUE, 2000)
  rna <- pp_one_omics(grid$rna, FALSE, 2000)
  
  for (namei in names(atac@scale.data)) {
    rna@scale.unshared.data[[namei]] <- t(atac@scale.data[[namei]])
    rna@var.unshared.features[[namei]] <- colnames(atac@scale.data[[namei]])
  }
  
  # browser()
  
  res_liger <- optimizeALS(
    rna, k=K, use.unshared = TRUE,
    max.iters = 30, thresh = 1e-10, rand.seed = seed,
    # remove.missing = FALSE
  )
  res_liger <- quantile_norm(res_liger)
  H <- as.data.frame.matrix(res_liger@H.norm)
  
  
  # 把顺序重新调换过来
  blabel <- sapply(strsplit(row.names(H), "-"), function(x) x[1])
  Hs <- list()
  for (bi in 1:5) {
    bi <- paste0("batch", bi)
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
  data_fn <- file.path(data_dir, "mop5b_full.mmod")
  res_fn <- file.path(res_dir, paste0("mop5b_full_all_", seedi, ".csv"))
  print(res_fn)
  run(data_fn, res_fn, use_pseduo = TRUE, K = 30, seed = seedi)
}

# # 虽然seed不会对模型没有影响，但是会影响subsample出来的数据集
# data_fns <- c()
# for (data_fn in list.files(data_dir)) {
#   if (grepl("pbmc_graph_feats_[0-9]*?_[0-9].mmod", data_fn)) {
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
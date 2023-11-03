# 1. 设置环境
renv::activate(project = normalizePath("../../../environments/env_stabmap/"))

# 2. 载入包和python函数
library(StabMap)
library(SingleCellMultiModal)
library(scran)
library(MultiAssayExperiment)
library(scater)

library(reticulate)
use_condaenv("mosaic", conda = "~/mambaforge/bin/conda")
mmAAVI <- import_from_path("mmAAVI", normalizePath("../../../src/"))



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
  for (bi in dat$batch_names) {
    grid[[bi]] <- list()

    atac <- dat$X[bi, "atac"]
    rna <- dat$X[bi, "rna"]
    protein <- dat$X[bi, "protein"]

    if (use_pseduo) {
      if ((!is.null(atac)) & is.null(rna)) {
        new_rna <- atac %*% net
        rna <- new_rna
      }
    }

    if (!is.null(atac)) {
      rownames(atac) <- paste0(bi, "_", 1:nrow(atac))
      colnames(atac) <- paste0("atac_", 1:ncol(atac))
    }
    if (!is.null(rna)) {
      rownames(rna) <- paste0(bi, "_", 1:nrow(rna))
      colnames(rna) <- paste0("rna_", 1:ncol(rna))
    }
    if (!is.null(protein)) {
      rownames(protein) <- paste0(bi, "_", 1:nrow(protein))
      colnames(protein) <- paste0("protein_", 1:ncol(protein))
    }
    grid[[bi]][["atac"]] <- atac
    grid[[bi]][["rna"]] <- rna
    grid[[bi]][["protein"]] <- protein
  }

  # 4. preprocessing
  # 将每个组学连接到一起，进行预处理，然后再分开
  grid_pp <- list()
  for (omic_name_i in c("atac", "rna", "protein")) {
    # print(paste0("preprocess ", omic_name_i, " ..."))
    grid_omic_i <- list()
    for (batch_name_i in names(grid)) {
      dati <- grid[[batch_name_i]][[omic_name_i]]
      if (!is.null(dati)) {grid_omic_i[[batch_name_i]] <- dati}
    }
    grid_omic_i_bind <- do.call(rbind, grid_omic_i)

    se_i <- SummarizedExperiment(list(counts = t(grid_omic_i_bind)))
    se_i <- logNormCounts(se_i)
    decomp <- modelGeneVar(se_i)
    hvgs <- decomp$mean>0.01 & decomp$p.value <= 0.1
    # print(sum(hvgs))
    se_i <- se_i[hvgs,]

    # 重新分开
    new_grid_i <- list()
    start <- 1
    for (name_i in names(grid_omic_i)) {
      dati <- grid_omic_i[[name_i]]
      if (is.null(dati)) next
      nc <- nrow(dati)
      end <- start + nc - 1
      new_grid_i[[name_i]] <- t(assays(se_i[, start:end])[["logcounts"]])
      start <- end + 1
    }

    grid_pp[[omic_name_i]] <- new_grid_i
  }
  # 重新组合，其中外层是batch，内部是多个组学的拼接，cell x gene -> gene x cell
  grid_pp_t <- list()
  for (batch_name_i in names(grid)) {
    grid_omic_i <- list()
    for (omics_name_i in c("atac", "rna", "protein")) {
      dati <- grid_pp[[omics_name_i]][[batch_name_i]]
      grid_omic_i[[omics_name_i]] <- dati
    }
    grid_omic_i <- do.call(cbind, grid_omic_i)
    grid_pp_t[[batch_name_i]] <- t(grid_omic_i)
  }
  lapply(grid_pp_t, dim)

  # 5. running stabmap
  stab <- stabMap(
    grid_pp_t, reference_list = c("batch1"),
    plot = FALSE,
    ncomponentsReference = K, ncomponentsSubset = K,
    # scale.center = FALSE, scale.scale = FALSE
  )

  # 6. saving results
  stab <- as.data.frame(stab)
  write.csv(stab, file = res_fn)
}


# 2. 读入数据
data_dir <- normalizePath("../res/1_pp/")
res_dir <- normalizePath("../res/3_comparison/stabmap/")
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

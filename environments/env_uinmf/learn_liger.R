library(rliger)

dat_dir <- "~/Documents/mosaic-GAN/data/BMMC_liger/"
D5T1 <- readRDS(paste0(dat_dir, 'GSM4138888_scATAC_BMMC_D5T1.RDS'))
rna1 <- readRDS(paste0(dat_dir, 'GSM4138872_scRNA_BMMC_D1T1.rds'))
rna2 <- readRDS(paste0(dat_dir, 'GSM4138873_scRNA_BMMC_D1T2.rds'))
bmmc.rna <- cbind(rna1,rna2)
rm(rna1, rna2)
print(c(dim(bmmc.rna), dim(D5T1)))

# 查看一下是否存在相同的dimnames
# 1. cell names (0)
intersect(colnames(bmmc.rna), colnames(D5T1))
# 2. variable names (18454)
length(intersect(rownames(bmmc.rna), rownames(D5T1)))
# 3. 各自多出的variable names
rna_only <- setdiff(rownames(bmmc.rna), rownames(D5T1))
length(rna_only)
print(rna_only[1:5])
atac_only <- setdiff(rownames(D5T1), rownames(bmmc.rna))
length(atac_only)
print(atac_only[1:5])
# 4. 查看variable names中是否分成genes和promoters两部分
#   promoters和genes的名称是一样的，只是在位置上promoters会比对应的genes
#   更靠后一些（在负链上则是更靠前一些）
genes <- read.table(paste0(dat_dir, "hg19_genes.bed"), sep = "\t")
promoters <- read.table(paste0(dat_dir, "hg19_promoters.bed"), sep = "\t")
print(dim(genes))
print(dim(promoters))
# 总结：这里atac其实是每个genes及其上游的promoters覆盖的区域内atac的总计数。

bmmc.data <- list(atac = D5T1, rna = bmmc.rna)
int.bmmc <- createLiger(bmmc.data)
int.bmmc <- normalize(int.bmmc)
int.bmmc <- selectGenes(int.bmmc, datasets.use = 2)
int.bmmc <- scaleNotCenter(int.bmmc)

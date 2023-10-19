dat_72 <- readRDS("./raw/scRNA_4138872.rds")
writeMM(dat_72, "./raw/scRNA_4138872.mtx")
write(colnames(dat_72), "./raw/scRNA_4138872_col.txt")
write(rownames(dat_72), "./raw/scRNA_4138872_row.txt")

dat_73 <- readRDS("./raw/scRNA_4138873.rds")
writeMM(dat_73, "./raw/scRNA_4138873.mtx")
write(colnames(dat_73), "./raw/scRNA_4138873_col.txt")
write(rownames(dat_73), "./raw/scRNA_4138873_row.txt")

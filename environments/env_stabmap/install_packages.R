renv::init(
  bare = TRUE,  # 载入环境时不会自动安装包
  bioconductor = TRUE  # 环境可以管理bioconductor上的包，即可以使用BiocManager进行安装
)

options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN"),
        BioC_mirror="https://mirrors.tuna.tsinghua.edu.cn/bioconductor")
install.packages("BiocManager")
BiocManager::install(c("scran", "SingleCellMultiModal", "scater"))

install.packages("devtools")
devtools::install_github("MarioniLab/StabMap")

# 与python之间的交互
install.packages("reticulate")

renv::snapshot()  # 保存当前的环境状态

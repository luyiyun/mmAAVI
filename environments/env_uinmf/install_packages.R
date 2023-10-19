renv::init(
  bare = TRUE,  # 载入环境时不会自动安装包
  bioconductor = TRUE  # 环境可以管理bioconductor上的包，即可以使用BiocManager进行安装
)
install.packages("devtools")
devtools::install_github('welch-lab/liger')  # 需要使用github上的最新版本才有UINMF的实现
install.packages("Matrix")  # 需要>1.5版本的Matrix
install.packages("Seurat")
# install.packages("BiocManager")
# BiocManager::install(c("Rhdf5lib", "rhdf5filters", "rhdf5"))
install.packages("reticulate")
renv::snapshot()  # 保存当前的环境状态
# 关于leiden的错误，需要在conda的base环境中安装pandas来解决
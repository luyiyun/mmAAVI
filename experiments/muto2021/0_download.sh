save_dir="./data"
mkdir -p $save_dir
cd $save_dir
wget http://download.gao-lab.org/GLUE/dataset/Muto-2021-RNA.h5ad
wget http://download.gao-lab.org/GLUE/dataset/Muto-2021-ATAC.h5ad
cd -
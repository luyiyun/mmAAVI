save_dir="./data"
mkdir -p $save_dir
cd $save_dir
wget http://download.gao-lab.org/GLUE/dataset/10x-ATAC-Brain5k.h5ad
wget http://download.gao-lab.org/GLUE/dataset/Luo-2017.h5ad
wget http://download.gao-lab.org/GLUE/dataset/Saunders-2018.h5ad
cd -

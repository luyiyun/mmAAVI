set -v

# conda install python=3.10  -y
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# # conda install -c rapidsai -c conda-forge -c nvidia  \
# #     rapids=23.02 python=3.10 cudatoolkit=11.8 -y
# conda install -c conda-forge -y \
#     ipdb flake8 jupyterlab=3.6.4 numpy pandas scipy networkx matplotlib seaborn \
#     scikit-learn umap-learn tqdm tensorboard hydra-core h5py pytables
# # 安装pyg
# conda install pyg pytorch-cluster pytorch-sparse -c pyg -y
# #
# #
# pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
# pip install biothings_client requests-cache
#
# # pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
#
# pip install GEOparse
# pip install scikit-misc  # for sc.pp.highly_variance_genes(flavor="seurat_v3")
#
# # 可以进行测试
# pip install pytest
# # 对jupyter进行版本控制
# # pip install jupytext


# --------------------------------------------------

mamba install -c rapidsai -c conda-forge -c nvidia rapids=23.02 python=3.10 cudatoolkit=11.8 -y
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
mamba install -c conda-forge -c nvidia -y \
    ipdb flake8 jupyterlab numpy pandas scipy networkx matplotlib seaborn scikit-learn umap-learn \
    tqdm tensorboard hydra-core h5py pytables scikit-misc jupytext pytest \
    anndata chex jax jaxlib=*=*cuda* cuda-nvcc scanpy rich pynndescent "igraph>0.9.0" plottable leidenalg # 这是scib-metrics的依赖

# 安装pyg
mamba install pyg pytorch-cluster pytorch-sparse pytorch-scatter -c pyg -y

pip install scib-metrics --no-deps
# pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
pip install biothings_client requests-cache GEOparse adjustText

# --------------------------------------------------

# # conda install pybedtools bedtools htslib -c conda-forge -c bioconda -y
# # conda install anndata -c conda-forge -y
# # pip install pytorch-lightning torchmetrics
# # conda install kaggle -c conda-forge -y
# # conda install nvitop -c conda-forge -y
# # pip install hdf5plugin
# pip install leidenalg igraph
# # pip install combat --no-deps
# # pip install mpmath
# # 为了tqdm在notebook中正常使用
# # pip install -U jupyterlab-widgets==1.1.1
# # pip install -U ipywidgets==7.7.2
#

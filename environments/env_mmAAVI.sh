set -v

mamba install -c rapidsai -c conda-forge -c nvidia rapids=23.02 python=3.10 cudatoolkit=11.8 -y
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
mamba install -c conda-forge -c nvidia -y \
    ipdb flake8 jupyterlab numpy pandas scipy networkx matplotlib seaborn scikit-learn umap-learn \
    tqdm tensorboard hydra-core h5py pytables scikit-misc jupytext pytest \
    anndata chex jax jaxlib=*=*cuda* cuda-nvcc scanpy rich pynndescent "igraph>0.9.0" plottable leidenalg # 这是scib-metrics的依赖

pip install scib-metrics --no-deps
pip install biothings_client requests-cache GEOparse adjustText

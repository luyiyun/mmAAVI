# conda create -n compar-multimap python=3.8 -y
# conda activate compar-multimap
mamba install python=3.10 scipy=1.8.0 h5py pytables scanpy python-igraph leidenalg -c conda-forge -y
pip install git+https://github.com/Teichlab/MultiMAP.git

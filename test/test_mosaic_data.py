import pytest
import numpy as np

from mmAAVI.dataset import DataGrid, MosaicData


def get_data_grid():
    grid = DataGrid([("b1", "o1", np.random.rand(10, 3)),
                     ("b2", "o2", np.random.rand(5, 4)),
                     ("b3", "o3", (3, 2))],
                    batch_names=["b1", "b2", "b3"],
                    omics_names=["o1", "o2", "o3"],)
    return grid


def get_mosaic_data():
    grid = get_data_grid()
    mdata = MosaicData(grid)
    return mdata


class TestMosaicData:

    def test_init(self):
        mdata = get_mosaic_data()
        print(mdata)

    def test_set_obs_var_column(self):
        mdata = get_mosaic_data()
        mdata.obs["a"] = 1.
        mdata.var["b"] = 2.
        print(mdata)

    def test_nets(self):
        mdata = get_mosaic_data()
        mdata.nets["net1"] = [("o1", "o2", np.random.rand(3, 4))]
        print(mdata)
        mdata.nets["net2"] = [("o1", "o2", np.random.rand(3, 4))]
        print(mdata)

    def test_reps(self):
        mdata = get_mosaic_data()
        mdata.reps["rep_neg"] = mdata.X.apply(lambda x: x if x is None else -x)
        assert mdata.reps["rep_neg"].batch_dims == mdata.X.batch_dims
        print(mdata)

    def test_to_anndata(self):
        mdata = get_mosaic_data()
        adata = mdata.to_anndata(False)
        assert adata.shape == (18, 9)
        print(adata)

    def test_reps_set_names_dims(self):
        mdata = get_mosaic_data()
        mdata.reps["rep_1"] = [
            ("b1", "o1", np.ones((10, 1))),
            ("b2", "o2", np.ones((5, 1))),
            ("b3", "o3", np.ones((3, 1))),
        ]
        with pytest.raises(AssertionError, match="the batch_dims_dict of "):
            mdata.reps["rep_2"] = [
                ("b1", "o1", np.ones((3, 1))),
                ("b2", "o2", np.ones((5, 1))),
                ("b3", "o3", np.ones((3, 1))),
            ]
        with pytest.raises(AssertionError, match="the omics_names of "):
            mdata.reps["rep_3"] = [
                ("b1", "oo", np.ones((10, 1))),
                ("b2", "o2", np.ones((5, 1))),
                ("b3", "o3", np.ones((3, 1))),
            ]

    def test_select_rows(self):
        mdata = get_mosaic_data()
        ind = [0, 1, 2, 9, 10, 12, 14, 15, 16, 17]
        mdatai = mdata.select_rows(ind)
        assert (mdatai.obs.index == ind).all()
        assert (mdatai.obs._batch[:4] == "b1").all()
        assert (mdatai.obs._batch[4:7] == "b2").all()
        assert (mdatai.obs._batch[7:] == "b3").all()

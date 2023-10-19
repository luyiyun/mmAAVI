import pytest
import numpy as np
import h5py

from mmAAVI.dataset import DataGrid


def get_data_grid():
    grid = DataGrid([("b1", "o1", np.random.rand(10, 3)),
                     ("b2", "o2", np.random.rand(5, 4)),
                     ("b3", "o3", (3, 2))],
                    batch_names=["b1", "b2", "b3"],
                    omics_names=["o1", "o2", "o3"],)
    return grid


class TestDataGrid:

    def test_init_seq_record(self):
        grid = get_data_grid()
        print(grid)

    def test_init_not_shared(self):
        # 修改一个bug
        grid = DataGrid([
            ("b1", "o1", np.random.rand(10, 3)),
            ("b2", "o1", np.random.rand(11, 3)),
            ("b3", "o1", np.random.rand(12, 3)),
            ("b4", "o2", np.random.rand(5, 4)),
            ("b5", "o2", np.random.rand(6, 4)),
            ("b6", "o2", np.random.rand(7, 4)),
        ])
        print(grid)

    def test_set_names(self):
        grid = get_data_grid()
        grid.batch_names = ["batch1", "batch2", "batch3"]
        grid.omics_names = ["omic1", "omic2", "omic3"]
        print(grid)

    def test_save_load(self, tmp_path):
        grid = get_data_grid()
        fn = tmp_path / "tmp.h5"
        with h5py.File(fn, "w") as f:
            grid.save_in_h5group(f)

        with h5py.File(fn, "r") as f:
            new_grid = DataGrid.load_from_h5group(f)

        assert grid.batch_dims_dict == new_grid.batch_dims_dict
        assert grid.omics_dims_dict == new_grid.omics_dims_dict
        assert str(grid) == str(new_grid)

    def test_set_null_for_zero_block(self):
        grid = DataGrid([("b1", "o1", np.random.rand(10, 3)),
                         ("b2", "o2", np.random.rand(5, 4)),
                         ("b3", "o3", np.zeros((3, 2)))],
                        batch_names=["b1", "b2", "b3"],
                        omics_names=["o1", "o2", "o3"],)
        grid.set_null_for_zero_block()
        assert grid["b3", "o3"] is None
        assert grid.batch_dims[2] == 3
        assert grid.omics_dims[2] == 2

        grid.set_null_for_zero_block(zero_dim=True)
        assert grid.batch_dims[2] == 0
        assert grid.omics_dims[2] == 0

    def test_diff_num_batch_omics(self):
        grid = DataGrid([("b1", "o1", np.random.rand(10, 3)),
                         ("b2", "o3", np.random.rand(5, 4))],
                        batch_names=["b1", "b2", "b3"],
                        omics_names=["o1", "o2", "o3"],)
        print(grid)

        with pytest.raises(AssertionError, match="batch dims is not equal"):
            grid = DataGrid([("b1", "o1", np.random.rand(10, 3)),
                             ("b1", "o3", np.random.rand(5, 4))],
                            batch_names=["b1", "b2", "b3"],
                            omics_names=["o1", "o2", "o3"],)

        with pytest.raises(AssertionError, match="omics dims is not equal"):
            grid = DataGrid([("b1", "o1", np.random.rand(10, 3)),
                             ("b2", "o1", np.random.rand(5, 4))],
                            batch_names=["b1", "b2", "b3"],
                            omics_names=["o1", "o2", "o3"],)

# print(grid["batch1"])
# print(grid["batch1", :])
# print(grid["batch1", "omic1"])
# print(grid[:, "omic1"])
# print(grid[0, 0])
# print(grid["batch2", 1])
# print(grid[0, :])
# print(grid[0, :, :])
# print(grid[0, :, :, :])
# print(grid[[0], [0, 1]])

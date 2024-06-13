r"""
Genomics operations
Refer from https://github.com/gao-lab/GLUE/blob/master/scglue/genomics.py
"""
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from biothings_client import get_client
from scipy import sparse as sp


class ConstrainedDataFrame(pd.DataFrame):
    r"""
    Data frame with certain format constraints

    Note
    ----
    Format constraints are checked and maintained automatically.
    """

    def __init__(self, *args, **kwargs) -> None:
        df = pd.DataFrame(*args, **kwargs)
        df = self.rectify(df)
        self.verify(df)
        super().__init__(df)

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self.verify(self)

    @property
    def _constructor(self) -> type:
        return type(self)

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:
        r"""
        Rectify data frame for format integrity

        Parameters
        ----------
        df
            Data frame to be rectified

        Returns
        -------
        rectified_df
            Rectified data frame
        """
        return df

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        r"""
        Verify data frame for format integrity

        Parameters
        ----------
        df
            Data frame to be verified
        """

    @property
    def df(self) -> pd.DataFrame:
        r"""
        Convert to regular data frame
        """
        return pd.DataFrame(self)

    def __repr__(self) -> str:
        r"""
        Note
        ----
        We need to explicitly call :func:`repr` on the regular data frame
        to bypass integrity verification, because when the terminal is
        too narrow, :mod:`pandas` would split the data frame internally,
        causing format verification to fail.
        """
        return repr(self.df)


class Bed(ConstrainedDataFrame):
    r"""
    BED format data frame
    """

    COLUMNS = pd.Index(
        [
            "chrom",
            "chromStart",
            "chromEnd",
            "name",
            "score",
            "strand",
            "thickStart",
            "thickEnd",
            "itemRgb",
            "blockCount",
            "blockSizes",
            "blockStarts",
        ]
    )

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = super(Bed, cls).rectify(df)
        COLUMNS = cls.COLUMNS.copy(deep=True)
        for item in COLUMNS:
            if item in df:
                if item in ("chromStart", "chromEnd"):
                    df[item] = df[item].astype(int)
                else:
                    df[item] = df[item].astype(str)
            elif item not in ("chrom", "chromStart", "chromEnd"):
                df[item] = "."
            else:
                raise ValueError(f"Required column {item} is missing!")
        return df.loc[:, COLUMNS]

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        super(Bed, cls).verify(df)
        if len(df.columns) != len(cls.COLUMNS) or np.any(
            df.columns != cls.COLUMNS
        ):
            raise ValueError("Invalid BED format!")

    @classmethod
    def read_bed(cls, fname: str) -> "Bed":
        r"""
        Read BED file

        Parameters
        ----------
        fname
            BED file

        Returns
        -------
        bed
            Loaded :class:`Bed` object
        """
        COLUMNS = cls.COLUMNS.copy(deep=True)
        loaded = pd.read_csv(fname, sep="\t", header=None, comment="#")
        loaded.columns = COLUMNS[: loaded.shape[1]]
        return cls(loaded)

    def write_bed(self, fname: str, ncols: Optional[int] = None) -> None:
        r"""
        Write BED file

        Parameters
        ----------
        fname
            BED file
        ncols
            Number of columns to write (by default write all columns)
        """
        if ncols and ncols < 3:
            raise ValueError("`ncols` must be larger than 3!")
        df = self.df.iloc[:, :ncols] if ncols else self
        df.to_csv(fname, sep="\t", header=False, index=False)

    def strand_specific_start_site(self) -> "Bed":
        r"""
        Convert to strand-specific start sites of genomic features

        Returns
        -------
        start_site_bed
            A new :class:`Bed` object, containing strand-specific start sites
            of the current :class:`Bed` object
        """
        if set(self["strand"]) != set(["+", "-"]):
            raise ValueError("Not all features are strand specific!")
        df = pd.DataFrame(self, copy=True)
        pos_strand = df.query("strand == '+'").index
        neg_strand = df.query("strand == '-'").index
        df.loc[pos_strand, "chromEnd"] = df.loc[pos_strand, "chromStart"] + 1
        df.loc[neg_strand, "chromStart"] = df.loc[neg_strand, "chromEnd"] - 1
        return type(self)(df)

    def strand_specific_end_site(self) -> "Bed":
        r"""
        Convert to strand-specific end sites of genomic features

        Returns
        -------
        end_site_bed
            A new :class:`Bed` object, containing strand-specific end sites
            of the current :class:`Bed` object
        """
        if set(self["strand"]) != set(["+", "-"]):
            raise ValueError("Not all features are strand specific!")
        df = pd.DataFrame(self, copy=True)
        pos_strand = df.query("strand == '+'").index
        neg_strand = df.query("strand == '-'").index
        df.loc[pos_strand, "chromStart"] = df.loc[pos_strand, "chromEnd"] - 1
        df.loc[neg_strand, "chromEnd"] = df.loc[neg_strand, "chromStart"] + 1
        return type(self)(df)

    def expand(
        self,
        upstream: int,
        downstream: int,
        chr_len: Optional[Mapping[str, int]] = None,
    ) -> "Bed":
        r"""
        Expand genomic features towards upstream and downstream

        Parameters
        ----------
        upstream
            Number of bps to expand in the upstream direction
        downstream
            Number of bps to expand in the downstream direction
        chr_len
            Length of each chromosome

        Returns
        -------
        expanded_bed
            A new :class:`Bed` object, containing expanded features
            of the current :class:`Bed` object

        Note
        ----
        Starting position < 0 after expansion is always trimmed.
        Ending position exceeding chromosome length is trimed only if
        ``chr_len`` is specified.
        """
        if upstream == downstream == 0:
            return self
        df = pd.DataFrame(self, copy=True)
        if upstream == downstream:  # symmetric
            df["chromStart"] -= upstream
            df["chromEnd"] += downstream
        else:  # asymmetric
            if set(df["strand"]) != set(["+", "-"]):
                raise ValueError("Not all features are strand specific!")
            pos_strand = df.query("strand == '+'").index
            neg_strand = df.query("strand == '-'").index
            if upstream:
                df.loc[pos_strand, "chromStart"] -= upstream
                df.loc[neg_strand, "chromEnd"] += upstream
            if downstream:
                df.loc[pos_strand, "chromEnd"] += downstream
                df.loc[neg_strand, "chromStart"] -= downstream
        df["chromStart"] = np.maximum(df["chromStart"], 0)
        if chr_len:
            chr_len = df["chrom"].map(chr_len)
            df["chromEnd"] = np.minimum(df["chromEnd"], chr_len)
        return type(self)(df)

    def window_graph(
        self,
        right: pd.DataFrame,
        window_size: int,
        use_chrom: Optional[Sequence[str]] = None,
    ) -> sp.coo_array:
        r"""
        Construct a window graph (coo_array) between two sets of genomic
        features, where features pairs within a window size are connected.

        Parameters
        ----------
        left
            First feature set
        right
            Second feature set
        window_size
            Window size (in bp)

        Returns
        -------
        graph
            Window graph
        """
        chrom_all = list(
            set(self["chrom"].unique()).union(right["chrom"].unique())
        )

        left = self
        left = left.reset_index()
        right = right.reset_index()

        net, row_index, col_index = [], [], []
        for chri in chrom_all:
            lefti = left.query("chrom == '%s'" % chri)
            righti = right.query("chrom == '%s'" % chri)

            row_index.append(lefti.index.values)
            col_index.append(righti.index.values)

            if use_chrom is not None and chri not in use_chrom:
                net.append(sp.coo_matrix((lefti.shape[0], righti.shape[0])))
                continue

            if lefti.shape[0] == 0 or righti.shape[0] == 0:
                net.append(sp.coo_matrix((lefti.shape[0], righti.shape[0])))
                continue

            xstart, xstop = (
                lefti["chromStart"].values,
                lefti["chromEnd"].values,
            )
            ystart, ystop = (
                righti["chromStart"].values,
                righti["chromEnd"].values,
            )
            ye_xs = ystop - xstart[:, None]
            xe_ys = xstop[:, None] - ystart

            mask = np.logical_and(ye_xs > 0, xe_ys > 0)
            if window_size == 0:
                net.append(sp.coo_matrix(mask.astype(float)))
                continue

            dist = np.full((lefti.shape[0], righti.shape[0]), np.NaN)
            dist[mask] = 0.0
            mask = xe_ys <= 0
            dist[mask] = xe_ys[mask] - 1
            mask = ye_xs <= 0
            dist[mask] = 1 - ye_xs[mask]
            neti = np.logical_and(dist >= -window_size, dist <= window_size)
            net.append(sp.coo_matrix(neti.astype(float)))

        net = sp.block_diag(net)
        row_index = np.concatenate(row_index)
        col_index = np.concatenate(col_index)

        # reorder
        leftind = pd.Series(range(left.shape[0]), index=left.index)
        leftind = leftind.loc[row_index].values
        rightind = pd.Series(range(right.shape[0]), index=right.index)
        rightind = rightind.loc[col_index].values
        net = sp.coo_array(
            (net.data, (leftind[net.row], rightind[net.col])), shape=net.shape
        )

        return net


def search_genomic_pos(
    genes: Sequence[str],
    cache_url: Optional[str] = None,
    remove_genes: Sequence[Tuple[str, float]] = (),
    species: str = "human",
) -> pd.DataFrame:
    chroms = [str(i) for i in range(1, 23)] + ["X", "Y"]

    gene = get_client("gene")
    gene.set_caching(cache_url)
    res = gene.querymany(
        genes,
        species=species,
        scopes=["symbol"],
        fields=[
            "_score",
            "name",
            "genomic_pos.chr",
            "genomic_pos.start",
            "genomic_pos.end",
            "genomic_pos.strand",
        ],
        as_dataframe=True,
        df_index=True,
    )
    # remove genes which not in autosomes and sex chromosomes
    res = res[res["genomic_pos.chr"].isin(chroms)]
    # remove two duplicated genes, remove manually
    if remove_genes:
        masks = []
        for namei, starti in remove_genes:
            maski = np.logical_and(
                res.index == namei, res["genomic_pos.start"] == starti
            )
            masks.append(maski)
        masks = np.stack(masks, axis=1)
        masks = np.any(masks, axis=1)
        res = res[~masks]
    # modify the var names
    res.rename(
        inplace=True,
        columns={
            "_score": "score",
            "genomic_pos.chr": "chrom",
            "genomic_pos.start": "chromStart",
            "genomic_pos.end": "chromEnd",
            "genomic_pos.strand": "strand",
        },
    )
    dfi = res.reindex(index=genes)
    gene.stop_caching()

    return dfi

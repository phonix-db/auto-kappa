"""
Tests for auto_kappa/io/kl.py: read_kl_file().

ANPHON writes .kl with fixed-width columns, so an overflowing component merges
with its neighbors. Such a line means the calculation failed, and it must not
be read (GitHub issue #15).
"""
import numpy as np

from auto_kappa.io.kl import KL

HEADER = "# Temperature [K], Thermal Conductivity (xx, xy, xz, yx, yy, yz, zx, zy, zz) [W/mK]\n"

LINE_100K = ("    100.00       150.1234         0.0000         0.0000         0.0000"
             "       150.1234         0.0000         0.0000         0.0000       140.5678\n")

# mp-23486-2, RbNiCl3.kl at 300 K
LINE_300K_MERGED = ("    300.003561674089978453714534400.0000-304606912785.4122"
                    "-980209081972535.8750-304606912785.41223561674088396586657775616.0000"
                    "440163170033648.1250-980209081972535.8750440163170033648.1250"
                    "   1052501.4184\n")


def test_merged_line_is_not_read(tmp_path):
    path = tmp_path / "RbNiCl3.kl"
    path.write_text(HEADER + LINE_100K + LINE_300K_MERGED)
    kl = KL(str(path))
    assert len(kl.data) == 1
    assert np.isclose(kl.get_kappa(100)['kxx'], 150.1234)
    assert kl.get_kappa(300) is None

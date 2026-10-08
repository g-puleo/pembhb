import numpy as np

from pembhb.sampler import lMcq_m1m2
from pembhb.utils import chirp_mass_from_m1m2


def test_lMcq_to_m1m2_roundtrip():
    m1 = np.array([3.0e6, 4.0e6, 5.0e6])
    m2 = np.array([1.0e6, 3.0e6, 5.0e6])
    x = np.stack([np.log10(chirp_mass_from_m1m2(m1, m2)), m1 / m2], axis=0)
    m1_rec, m2_rec = lMcq_m1m2(x)
    np.testing.assert_allclose(m1_rec, m1, rtol=1e-12)
    np.testing.assert_allclose(m2_rec, m2, rtol=1e-12)

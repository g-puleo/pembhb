import os

import pytest

from pembhb import ROOT_DIR
from pembhb.utils import marginal_names_to_indices, read_config, resolve_marginal_names


def test_names_map_to_basis_indices():
    m = {"f": [["logMchirp"], ["chi_eff"], ["lambda", "sinbeta"], ["Deltat"]]}
    assert marginal_names_to_indices(m, "chieff_chidiff") == {"f": [[0], [2], [7, 8], [10]]}


def test_duplicate_name_rejected():
    with pytest.raises(ValueError, match="two marginals"):
        marginal_names_to_indices({"f": [["lambda"], ["lambda", "sinbeta"]]}, "chieff_chidiff")


def test_unknown_name_and_indices_rejected():
    with pytest.raises(ValueError, match="unknown"):
        marginal_names_to_indices({"f": [["chi1"]]}, "chieff_chidiff")
    with pytest.raises(TypeError):
        marginal_names_to_indices({"f": [[0]]}, "chieff_chidiff")


def test_shipped_train_config_resolves():
    tc = read_config(os.path.join(ROOT_DIR, "configs", "train_config.yaml"))
    dc = read_config(os.path.join(ROOT_DIR, "configs", "datagen_config.yaml"))
    resolve_marginal_names(tc, dc["spin_param_basis"])
    flat = [i for m in tc["marginals"]["f"] for i in m]
    assert sorted(flat) == list(range(11))

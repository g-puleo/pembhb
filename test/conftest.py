import pytest

import pembhb


@pytest.fixture(autouse=True)
def _reset_precision():
    # Precision is process-global; stop one test's choice leaking into the next.
    pembhb.set_precision("float32")
    yield

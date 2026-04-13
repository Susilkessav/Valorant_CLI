from __future__ import annotations

import pytest

from valocoach.core.config import clear_settings_cache


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    clear_settings_cache()
    yield
    clear_settings_cache()

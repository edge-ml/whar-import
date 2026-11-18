"""Checking the settings of an import (pure: no whar_datasets needed)."""
import sys
import types

import pytest

# the router imports the whar wrapper, which imports whar_datasets; the
# function under test does not need it
sys.modules.setdefault("app.whar", types.ModuleType("app.whar"))
from app.routers.whar import resolve_options  # noqa: E402

INFO = {"id": "WISDM", "channels": ["acc_x", "acc_y", "acc_z"], "activities": ["walking", "jogging", "sitting"]}


def test_nothing_chosen_means_everything():
    assert resolve_options({"dataset_id": "WISDM"}, INFO) == {"channels": None, "activities": None, "sampling_freq": None}


def test_choices():
    out = resolve_options({"channels": ["acc_x"], "activities": ["walking", "sitting"], "sampling_freq": "25"}, INFO)
    assert out == {"channels": ["acc_x"], "activities": ["walking", "sitting"], "sampling_freq": 25.0}
    # all of them is the same as no choice
    assert resolve_options({"channels": ["acc_z", "acc_x", "acc_y"]}, INFO)["channels"] is None


@pytest.mark.parametrize("body, reason", [
    ({"channels": []}, "at least one"),
    ({"channels": "acc_x"}, "list of names"),
    ({"channels": ["mag_x"]}, "no channels mag_x"),
    ({"activities": ["flying"]}, "no activities flying"),
    ({"sampling_freq": 0}, "between"),
    ({"sampling_freq": 5000}, "between"),
    ({"sampling_freq": "fast"}, "a number"),
])
def test_refused(body, reason):
    with pytest.raises(ValueError, match=reason):
        resolve_options(body, INFO)

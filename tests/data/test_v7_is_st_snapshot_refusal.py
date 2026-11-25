"""Audit F1: pre-05-19 silver is_st rows carry `tickflow:exchanges.get_instruments`
(a current-name snapshot painted over history); the builder must refuse is_st."""

import pandas as pd

from quantagent.data.dataset_builder.v7_training_dataset import _refused_feature_columns


def test_instrument_snapshot_provenance_is_refused_as_a_feature() -> None:
    frame = pd.DataFrame({"is_st": [True, False],
                          "is_st_provenance": ["tickflow:exchanges.get_instruments",
                                               "dated_pit_st_evidence"]})
    assert _refused_feature_columns(frame, ["is_st"]) == {
        "is_st": "current_snapshot_broadcast_flag"}
    dated = frame.assign(is_st_provenance="dated_pit_st_evidence")
    assert _refused_feature_columns(dated, ["is_st"]) == {}

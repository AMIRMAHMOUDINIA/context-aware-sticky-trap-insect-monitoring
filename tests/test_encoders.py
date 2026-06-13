import pandas as pd

from insect_context_ai.data.encoders import ContextEncoder, LabelEncoder


def test_label_encoder_is_deterministic():
    encoder = LabelEncoder.fit(["species_b", "species_a", "species_b"])
    assert encoder.class_to_idx == {"species_a": 0, "species_b": 1}
    assert encoder.transform(["species_b", "species_a"]) == [1, 0]


def test_context_encoder_reserves_zero_for_unknown():
    frame = pd.DataFrame(
        {
            "device": ["DSLR", "webcam"],
            "trap_color": ["yellow", "blue"],
        }
    )
    encoder = ContextEncoder.fit(frame, ["device", "trap_color"])
    encoded = encoder.transform_row(
        pd.Series({"device": "new_device", "trap_color": "yellow"}),
        ["device", "trap_color"],
    )
    assert encoded[0] == 0
    assert encoded[1] > 0

from src.data.generate_data import generate_dataset
from src.data.validate_data import validate_frame


def test_generated_datasets_are_valid():
    assert validate_frame(generate_dataset("v1", 300)) == []
    assert validate_frame(generate_dataset("v2", 300)) == []


def test_invalid_range_is_reported():
    frame = generate_dataset("v1", 300)
    frame.loc[0, "pottery_load_kg"] = -1
    assert any("pottery_load_kg" in error for error in validate_frame(frame))

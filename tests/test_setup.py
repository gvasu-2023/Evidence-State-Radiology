from pathlib import Path


def test_project_structure():
    root = Path(__file__).resolve().parents[1]

    required_dirs = [
        "data",
        "src",
        "models",
        "experiments",
        "results",
        "configs",
        "tests",
    ]

    for directory in required_dirs:
        assert (root / directory).exists(), f"Missing directory: {directory}"
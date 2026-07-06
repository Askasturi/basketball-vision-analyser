
from basketball_vision_analyser import __project_name__, __version__


def test_package_imports() -> None:
    assert __project_name__ == "Basketball Vision Analyser"
    assert __version__ == "0.1.0"

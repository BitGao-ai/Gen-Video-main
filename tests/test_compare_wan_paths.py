import importlib.util
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location(
    "compare_wan_paths", Path(__file__).resolve().parents[1] /
    "scripts/diagnose/compare_wan_paths.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_difference():
    x = torch.ones(2, 3)
    assert diagnostic.difference(x, x)["relative_l2"] == 0
    assert diagnostic.difference(x * 2, x)["mae"] == 1
    assert "mae" not in diagnostic.difference(x, x[:1])
    assert not diagnostic.difference(x * float("nan"), x)["finite"]


def test_contact_sheet(tmp_path):
    from PIL import Image
    path = tmp_path / "frames.png"
    diagnostic.contact_sheet(torch.ones(1, 3, 9, 16, 32), path)
    with Image.open(path) as image:
        assert image.size == (160, 16)
        assert image.getpixel((0, 0)) == (255, 255, 255)


def test_json_nonfinite_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        diagnostic.save_json(tmp_path / "bad.json", {"x": float("nan")})

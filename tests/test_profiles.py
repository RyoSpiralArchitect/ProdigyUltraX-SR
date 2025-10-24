import pytest

torch = pytest.importorskip("torch")

from ProdigyUltra_SR_SplineTempX10 import (
    _bspline_profile_1d,
    _eval_conv_row_channel_profile,
    _eval_spatial_profile_conv,
)


def test_eval_spatial_profile_conv_honors_normalized_flag():
    spec = {
        "min": 0.0,
        "max": 10.0,
        "kh": {
            "degree": 1,
            "ctrl": [0.0, 1.0],
            "knots": [0.0, 0.5, 1.5, 2.0],
            "normalized": False,
        },
        "kw": {"degree": 0, "ctrl": [1.0], "normalized": True},
    }
    result = _eval_spatial_profile_conv(3, 1, spec, depth_norm=0.0)
    expected = _bspline_profile_1d(
        3,
        1,
        [0.0, 1.0],
        knots=[0.0, 0.5, 1.5, 2.0],
        normalized=False,
    )
    torch.testing.assert_close(result.view(-1), expected.to(torch.float32))


def test_eval_conv_row_channel_profile_honors_normalized_flag():
    spec = {
        "min": 0.0,
        "max": 10.0,
        "out": {
            "degree": 1,
            "ctrl": [1.0, 2.0],
            "knots": [0.0, 0.5, 1.5, 2.0],
            "normalized": False,
        },
        "in": {
            "degree": 1,
            "ctrl": [1.0, 3.0],
            "knots": [0.0, 0.25, 0.75, 1.0],
            "normalized": False,
        },
        "depth": {
            "degree": 1,
            "ctrl": [1.0, 2.0],
            "knots": [0.0, 0.4, 0.8, 1.2],
            "normalized": False,
        },
    }
    result = _eval_conv_row_channel_profile(3, 2, spec, depth_norm=0.5)

    y_out = _bspline_profile_1d(
        3,
        1,
        [1.0, 2.0],
        knots=[0.0, 0.5, 1.5, 2.0],
        normalized=False,
    )
    y_in = _bspline_profile_1d(
        2,
        1,
        [1.0, 3.0],
        knots=[0.0, 0.25, 0.75, 1.0],
        normalized=False,
    )
    depth = _bspline_profile_1d(
        1,
        1,
        [1.0, 2.0],
        knots=[0.0, 0.4, 0.8, 1.2],
        normalized=False,
    )[0]
    expected = (y_out.view(-1, 1) * y_in.view(1, -1)).reshape(-1).to(torch.float32) * depth
    torch.testing.assert_close(result, expected)

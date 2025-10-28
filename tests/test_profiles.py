import pytest

torch = pytest.importorskip("torch")

from ProdigyUltra_SR_SplineTempX10 import (
    _bspline_basis_all,
    _bspline_domain_from_knots,
    _bspline_profile_1d,
    _eval_conv_row_channel_profile,
    _eval_spatial_profile_conv,
)


def test_bspline_profile_1d_non_normalized_knots():
    knots = [0.0, 0.5, 1.5, 2.0]
    ctrl = [0.0, 1.0]
    result = _bspline_profile_1d(5, 1, ctrl, knots=knots, normalized=False)
    knot_tensor = torch.tensor(knots, dtype=torch.float64)
    lo, hi = _bspline_domain_from_knots(knot_tensor, 1)
    xs = torch.linspace(lo, hi, steps=5, dtype=torch.float64)
    B = _bspline_basis_all(xs, 1, knot_tensor)
    expected = (B @ torch.tensor(ctrl, dtype=torch.float64).view(-1, 1)).squeeze(1).to(torch.float32)
    torch.testing.assert_close(result, expected)


def test_bspline_profile_1d_handles_nested_normalized_dict_with_key():
    knots = [0.0, 0.5, 1.5, 2.0]
    ctrl = [0.0, 1.0]
    normalized = {"kh": {"default": False}, "default": True}
    result = _bspline_profile_1d(4, 1, ctrl, knots=knots, normalized=normalized, key="kh")

    knot_tensor = torch.tensor(knots, dtype=torch.float64)
    lo, hi = _bspline_domain_from_knots(knot_tensor, 1)
    xs = torch.linspace(lo, hi, steps=4, dtype=torch.float64)
    basis = _bspline_basis_all(xs, 1, knot_tensor)
    expected = (basis @ torch.tensor(ctrl, dtype=torch.float64).view(-1, 1)).squeeze(1).to(torch.float32)
    torch.testing.assert_close(result, expected)


def test_bspline_profile_1d_handles_nested_default_normalized_dict():
    knots = [0.0, 0.5, 1.5, 2.0]
    ctrl = [0.0, 1.0]
    normalized = {"default": {"default": False}}
    result = _bspline_profile_1d(4, 1, ctrl, knots=knots, normalized=normalized, key="kw")

    knot_tensor = torch.tensor(knots, dtype=torch.float64)
    lo, hi = _bspline_domain_from_knots(knot_tensor, 1)
    xs = torch.linspace(lo, hi, steps=4, dtype=torch.float64)
    basis = _bspline_basis_all(xs, 1, knot_tensor)
    expected = (basis @ torch.tensor(ctrl, dtype=torch.float64).view(-1, 1)).squeeze(1).to(torch.float32)
    torch.testing.assert_close(result, expected)


def test_eval_spatial_profile_conv_honors_normalized_and_depth_norm():
    spec = {
        "min": 0.0,
        "max": 10.0,
        "depth": {
            "degree": 1,
            "ctrl": [1.0, 3.0],
            "knots": [0.0, 1.0, 2.0, 3.0],
            "normalized": False,
        },
        "combine": "mul",
        "kh": {
            "degree": 1,
            "ctrl": [0.0, 1.0],
            "knots": [0.0, 0.5, 1.5, 2.0],
            "normalized": False,
        },
        "kw": {"degree": 0, "ctrl": [2.0], "normalized": True},
    }
    result = _eval_spatial_profile_conv(3, 1, spec, depth_norm=0.5)

    depth_knots = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    depth_ctrl = torch.tensor([1.0, 3.0], dtype=torch.float64)
    depth_sample = torch.tensor([1.5], dtype=torch.float64)
    depth_basis = _bspline_basis_all(depth_sample, 1, depth_knots)
    depth_val = (depth_basis @ depth_ctrl.view(-1, 1)).squeeze(1).to(torch.float32)[0]

    kh_knots = torch.tensor([0.0, 0.5, 1.5, 2.0], dtype=torch.float64)
    kh_ctrl = torch.tensor([0.0, 1.0], dtype=torch.float64)
    xs_h = torch.linspace(0.0, 1.0, steps=3, dtype=torch.float64)
    lo_h, hi_h = _bspline_domain_from_knots(kh_knots, 1)
    xs_h = xs_h * (hi_h - lo_h) + lo_h
    kh_basis = _bspline_basis_all(xs_h, 1, kh_knots)
    kh_vals = (kh_basis @ kh_ctrl.view(-1, 1)).squeeze(1).to(torch.float32)

    kw_vals = torch.full((1,), 2.0, dtype=torch.float32)
    expected = (kh_vals.view(3, 1) * kw_vals.view(1, 1)).reshape(-1) * depth_val
    torch.testing.assert_close(result, expected.clamp(spec["min"], spec["max"]))


def test_eval_spatial_profile_conv_skips_rescale_for_absolute_depth_input():
    spec = {
        "min": 0.0,
        "max": 10.0,
        "depth": {
            "degree": 1,
            "ctrl": [1.0, 3.0],
            "knots": [0.0, 1.0, 2.0, 3.0],
            "normalized": False,
        },
        "combine": "mul",
        "kh": {"degree": 0, "ctrl": [1.0], "normalized": True},
        "kw": {"degree": 0, "ctrl": [2.0], "normalized": True},
    }
    depth_norm = 2.5  # already in absolute coordinates; should not be rescaled.
    result = _eval_spatial_profile_conv(1, 1, spec, depth_norm=depth_norm)

    depth_knots = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    depth_ctrl = torch.tensor([1.0, 3.0], dtype=torch.float64)
    depth_sample = torch.tensor([depth_norm], dtype=torch.float64)
    depth_basis = _bspline_basis_all(depth_sample, 1, depth_knots)
    depth_val = (depth_basis @ depth_ctrl.view(-1, 1)).squeeze(1).to(torch.float32)[0]

    expected = torch.tensor([2.0 * depth_val], dtype=torch.float32)
    torch.testing.assert_close(result, expected.clamp(spec["min"], spec["max"]))


def test_eval_conv_row_channel_profile_honors_normalized_and_depth_norm():
    spec = {
        "min": 0.0,
        "max": 10.0,
        "out": {
            "degree": 1,
            "ctrl": [1.0, 2.0],
            "knots": [0.0, 1.0, 3.0, 4.0],
            "normalized": False,
        },
        "in": {
            "degree": 1,
            "ctrl": [1.0, 3.0],
            "knots": [0.0, 0.25, 0.75, 1.5],
            "normalized": False,
        },
        "depth": {
            "degree": 1,
            "ctrl": [1.0, 4.0],
            "knots": [0.0, 0.5, 1.0, 1.5],
            "normalized": False,
        },
        "combine": "mul",
    }
    result = _eval_conv_row_channel_profile(3, 2, spec, depth_norm=0.25)

    out_knots = torch.tensor([0.0, 1.0, 3.0, 4.0], dtype=torch.float64)
    out_ctrl = torch.tensor([1.0, 2.0], dtype=torch.float64)
    lo_out, hi_out = _bspline_domain_from_knots(out_knots, 1)
    xs_out = torch.linspace(lo_out, hi_out, steps=3, dtype=torch.float64)
    out_vals = (_bspline_basis_all(xs_out, 1, out_knots) @ out_ctrl.view(-1, 1)).squeeze(1).to(torch.float32)

    in_knots = torch.tensor([0.0, 0.25, 0.75, 1.5], dtype=torch.float64)
    in_ctrl = torch.tensor([1.0, 3.0], dtype=torch.float64)
    lo_in, hi_in = _bspline_domain_from_knots(in_knots, 1)
    xs_in = torch.linspace(lo_in, hi_in, steps=2, dtype=torch.float64)
    in_vals = (_bspline_basis_all(xs_in, 1, in_knots) @ in_ctrl.view(-1, 1)).squeeze(1).to(torch.float32)

    depth_knots = torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float64)
    depth_ctrl = torch.tensor([1.0, 4.0], dtype=torch.float64)
    lo_d, hi_d = _bspline_domain_from_knots(depth_knots, 1)
    depth_sample = torch.tensor([lo_d + (hi_d - lo_d) * 0.25], dtype=torch.float64)
    depth_val = (_bspline_basis_all(depth_sample, 1, depth_knots) @ depth_ctrl.view(-1, 1)).squeeze(1).to(torch.float32)[0]

    expected = (out_vals.view(-1, 1) * in_vals.view(1, -1)).reshape(-1) * depth_val
    torch.testing.assert_close(result, expected.clamp(spec["min"], spec["max"]))


def test_eval_conv_row_channel_profile_skips_rescale_for_absolute_depth_input():
    spec = {
        "min": 0.0,
        "max": 10.0,
        "out": {"degree": 0, "ctrl": [1.5], "normalized": True},
        "in": {"degree": 0, "ctrl": [2.0], "normalized": True},
        "depth": {
            "degree": 1,
            "ctrl": [1.0, 4.0],
            "knots": [0.0, 0.5, 1.0, 1.5],
            "normalized": False,
        },
        "combine": "mul",
    }
    depth_norm = 1.25
    result = _eval_conv_row_channel_profile(1, 1, spec, depth_norm=depth_norm)

    depth_knots = torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float64)
    depth_ctrl = torch.tensor([1.0, 4.0], dtype=torch.float64)
    depth_sample = torch.tensor([depth_norm], dtype=torch.float64)
    depth_val = (
        _bspline_basis_all(depth_sample, 1, depth_knots) @ depth_ctrl.view(-1, 1)
    ).squeeze(1).to(torch.float32)[0]

    expected = torch.tensor([1.5 * 2.0 * depth_val], dtype=torch.float32)
    torch.testing.assert_close(result, expected.clamp(spec["min"], spec["max"]))


def test_eval_spatial_profile_conv_accepts_tensor_ctrl_and_normalized_dict():
    spec = {
        "min": 0.0,
        "max": 10.0,
        "combine": "mul",
        "depth": {
            "degree": 1,
            "ctrl": torch.tensor([1.0, 2.0]),
            "knots": torch.tensor([0.0, 1.0, 2.0, 3.0]),
            "normalized": {"default": False},
        },
        "kh": {"degree": 0, "ctrl": torch.tensor([1.5]), "normalized": {"default": True}},
        "kw": {"degree": 0, "ctrl": [2.0], "normalized": {"default": True}},
    }
    depth_norm = 2.5
    result = _eval_spatial_profile_conv(1, 1, spec, depth_norm)

    depth_knots = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    depth_ctrl = torch.tensor([1.0, 2.0], dtype=torch.float64)
    depth_sample = torch.tensor([depth_norm], dtype=torch.float64)
    depth_basis = _bspline_basis_all(depth_sample, 1, depth_knots)
    depth_val = (depth_basis @ depth_ctrl.view(-1, 1)).squeeze(1).to(torch.float32)[0]

    expected = torch.tensor([1.5 * 2.0], dtype=torch.float32) * depth_val
    torch.testing.assert_close(result, expected.clamp(spec["min"], spec["max"]))


def test_eval_spatial_profile_conv_supports_default_dict_ctrl_and_knots():
    spec = {
        "min": 0.0,
        "max": 10.0,
        "combine": "mul",
        "kh": {
            "degree": 1,
            "ctrl": {"default": [0.0, 1.0]},
            "knots": {"default": [0.0, 0.5, 1.5, 2.0]},
            "normalized": {"default": False},
        },
        "kw": {
            "degree": 0,
            "ctrl": {"default": [2.0]},
            "normalized": {"default": True},
        },
    }
    result = _eval_spatial_profile_conv(4, 2, spec, depth_norm=0.3)

    kh_knots = torch.tensor([0.0, 0.5, 1.5, 2.0], dtype=torch.float64)
    kh_ctrl = torch.tensor([0.0, 1.0], dtype=torch.float64)
    xs_h = torch.linspace(0.0, 1.0, steps=4, dtype=torch.float64)
    lo_h, hi_h = _bspline_domain_from_knots(kh_knots, 1)
    xs_h = xs_h * (hi_h - lo_h) + lo_h
    kh_vals = (
        _bspline_basis_all(xs_h, 1, kh_knots) @ kh_ctrl.view(-1, 1)
    ).squeeze(1).to(torch.float32)

    kw_vals = torch.full((2,), 2.0, dtype=torch.float32)
    expected = (kh_vals.view(4, 1) * kw_vals.view(1, 2)).reshape(-1)
    torch.testing.assert_close(result, expected.clamp(spec["min"], spec["max"]))


def test_eval_conv_row_channel_profile_supports_default_dict_depth_specs():
    spec = {
        "min": 0.0,
        "max": 5.0,
        "out": {"degree": 0, "ctrl": torch.tensor([1.0]), "normalized": True},
        "in": {"degree": 0, "ctrl": [2.0], "normalized": {"default": True}},
        "depth": {
            "degree": 1,
            "ctrl": {"default": torch.tensor([1.0, 2.0])},
            "knots": {"default": torch.tensor([0.0, 1.0, 2.0, 3.0])},
            "normalized": {"default": False},
        },
        "combine": "mul",
    }
    depth_norm = 2.5
    result = _eval_conv_row_channel_profile(1, 1, spec, depth_norm)

    out_vals = torch.tensor([1.0], dtype=torch.float32)
    in_vals = torch.tensor([2.0], dtype=torch.float32)
    depth_knots = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    depth_ctrl = torch.tensor([1.0, 2.0], dtype=torch.float64)
    depth_sample = torch.tensor([depth_norm], dtype=torch.float64)
    depth_basis = _bspline_basis_all(depth_sample, 1, depth_knots)
    depth_val = (depth_basis @ depth_ctrl.view(-1, 1)).squeeze(1).to(torch.float32)[0]

    expected = (out_vals.view(-1, 1) * in_vals.view(1, -1)).reshape(-1) * depth_val
    torch.testing.assert_close(result, expected.clamp(spec["min"], spec["max"]))


def test_eval_conv_row_channel_profile_accepts_ctrl_dicts():
    spec = {
        "min": 0.0,
        "max": 10.0,
        "combine": "mul",
        "out": {
            "degree": 1,
            "ctrl": {"default": [1.0, 2.0]},
            "knots": {"default": [0.0, 1.0, 3.0, 4.0]},
            "normalized": {"default": False},
        },
        "in": {
            "degree": 1,
            "ctrl": {"default": [1.0, 3.0]},
            "knots": {"default": [0.0, 0.25, 0.75, 1.5]},
            "normalized": {"default": False},
        },
    }

    result = _eval_conv_row_channel_profile(3, 2, spec, depth_norm=0.5)

    out_knots = torch.tensor([0.0, 1.0, 3.0, 4.0], dtype=torch.float64)
    out_ctrl = torch.tensor([1.0, 2.0], dtype=torch.float64)
    xs_out = torch.linspace(0.0, 1.0, steps=3, dtype=torch.float64)
    lo_out, hi_out = _bspline_domain_from_knots(out_knots, 1)
    xs_out = xs_out * (hi_out - lo_out) + lo_out
    out_vals = (
        _bspline_basis_all(xs_out, 1, out_knots) @ out_ctrl.view(-1, 1)
    ).squeeze(1).to(torch.float32)

    in_knots = torch.tensor([0.0, 0.25, 0.75, 1.5], dtype=torch.float64)
    in_ctrl = torch.tensor([1.0, 3.0], dtype=torch.float64)
    xs_in = torch.linspace(0.0, 1.0, steps=2, dtype=torch.float64)
    lo_in, hi_in = _bspline_domain_from_knots(in_knots, 1)
    xs_in = xs_in * (hi_in - lo_in) + lo_in
    in_vals = (
        _bspline_basis_all(xs_in, 1, in_knots) @ in_ctrl.view(-1, 1)
    ).squeeze(1).to(torch.float32)

    expected = (out_vals.view(-1, 1) * in_vals.view(1, -1)).reshape(-1)
    torch.testing.assert_close(result, expected.clamp(spec["min"], spec["max"]))

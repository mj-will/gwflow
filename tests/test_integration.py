import pytest
import torch

from gwflow.gwflow import GWCalFlow


@pytest.mark.parametrize("context_dim", [None, 20])
@pytest.mark.parametrize("calibration_model", ["nn", "gaussian"])
@pytest.mark.parametrize(
    "inputs",
    [
        {"gw_dim": 5, "cal_dim": 3},
        {"gw_idx": [0, 2, 4, 6, 7], "cal_idx": [1, 3, 5]},
        {
            "parameters": [
                "m1",
                "m2",
                "dist",
                "iota",
                "psi",
                "calib_1",
                "calib_2",
                "calib_3",
            ],
            "calib_regex": "calib_.*",
        },
    ],
)
def test_calflow(inputs, context_dim, calibration_model):
    flow = GWCalFlow(
        context_dim=context_dim,
        calibration_model=calibration_model,
        **inputs,
    )

    assert flow.gw_dim == 5
    assert flow.cal_dim == 3

    gw_dim = flow.gw_dim
    cal_dim = flow.cal_dim

    # --- prepare input data ---
    batch_size = 16
    if context_dim:
        context_size = 10
        c = torch.randn(context_size, context_dim)
        dist = flow(c)
    else:
        c = None
        dist = flow()

    # --- sampling ---
    samples = dist.sample((batch_size,))
    if context_dim:
        assert samples.shape == (batch_size, context_size, gw_dim + cal_dim)
    else:
        assert samples.shape == (batch_size, gw_dim + cal_dim)

    # --- log prob ---
    logp = dist.log_prob(samples)
    if context_dim:
        assert logp.shape == (batch_size, context_size)
    else:
        assert logp.shape == (batch_size,)
    assert torch.isfinite(logp).all()

    # --- sample + log prob ---
    samples, logp = dist.rsample_and_log_prob((batch_size,))
    if context_dim:
        assert samples.shape == (batch_size, context_size, gw_dim + cal_dim)
        assert logp.shape == (batch_size, context_size)
    else:
        assert samples.shape == (batch_size, gw_dim + cal_dim)
        assert logp.shape == (batch_size,)
    assert torch.isfinite(logp).all()

    # --- training step (sanity check) ---
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = -logp.mean()
    loss.backward()
    optimizer.step()

    # parameters should have gradients
    grads = [p.grad for p in flow.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)

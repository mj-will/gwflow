# gwflow

Normalizing flows for gravitational-wave analyses.

`gwflow` builds on `zuko` and implements normalizing flows with problem-specific
modifications.

## Installation

`gwflow` is currently only available from source:

```
pip install git+git@github.com:mj-will/gwflow.git
```

## Flows

### `GWCalFlow`

This flow is designed for use in analyses where there are large number of
(simple) calibration parameters with approximately Gaussian distributions.

It approximates the distribution over the calibration parameters separately
to the main GW parameters. It supports two options for handling the
calibration parameters:

- `calibration_model='gaussian'`: fit a Gaussian with a diagonal covariance matrix
- `calibration_model='nn'`: fit a neural network that estimates the per-parameter mean and variance for of a diagonal Gaussian


### Usage

`GWCalFlow` supports three different ways of specifying which parameters
are 'GW' parameters and which are calibration parameters.

We provided a factory class than can be used to instantiate the flow using various
different methods:

```python
from gwflow import GWClaFlow
```

Then, assuming data lies in an n-dimensional space with parameters that
described the signal (GW parameters) and calibration parameters. The objects
returned in each support the same functionality as standard flows from
`zuko`.

In all cases, additional keyword arguments can be passed when instantiating
the class and these will be used to configure e.g. the normalizing flow.

**Parameters names**

Using a list of parameter names and regex:

```python
flow = GWCalFlow(
    parameters=["chirp_mass", "mass_ratio", ..., "recalib_H1_amplitude_0", ...],
    calib_regex=".*calib.*",
)
```

**Indices**

Using the indices for GW and calibration parameters

```python
flow = GWCalFlow(
    gw_idx=[0, 1, 3],
    cal_idx=[2, 5]
)
```

**Slices**

Using slices, assuming the GW parameters are first

```python
flow = GWCalFlow(
    gw_dim=15,
    cal_dim=20,
)
```

### Basic usage

Flows are used in the same way as `zuko` flows:

```python
flow = GWFlowClas(gw_dim=15, cal_dim=20, context=4, transforms=3)

x = flow().sample((10,))

log_prob = flow().log_prob(x)

# If the flow is conditional
x = flow(c).sample((10,))
```

For more details, see the [`zuko` documentation](https://zuko.readthedocs.io/stable/index.html)

**Note:** since the transform only applies to a subset of the parameter space,
the flow does not expose this attribute.


## Citation

If you use `gwflow` in your work, please cite our DOI (to be added) and [`zuko`](https://github.com/probabilists/zuko).

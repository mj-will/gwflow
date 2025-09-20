import inspect
import logging
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import zuko

logger = logging.getLogger(__name__)


class BaseGWCalFlow(nn.Module):
    """
    Base class for GW calibration normalizing flows.

    Parameters
    ----------
    gw_dim : int
        Dimensionality of GW parameters.
    cal_dim : int
        Dimensionality of calibration parameters.
    context_dim : Optional[int], optional
        Dimensionality of context vector, by default None.
    hidden_features : Sequence[int], optional
        Hidden layer sizes for GW flow network, by default (128, 128).
    cal_hidden_features : Sequence[int], optional
        Hidden layer sizes for calibration network, by default (128,).
    """

    def __init__(
        self,
        gw_dim: int,
        cal_dim: int,
        flow_class: str | Callable = "MAF",
        context_dim: Optional[int] = None,
        hidden_features: Sequence[int] = (128, 128),
        **kwargs,
    ) -> None:
        super().__init__()

        self.gw_dim = gw_dim
        self.cal_dim = cal_dim
        self.context_dim = context_dim

        # Configure calibration parameters
        cal_kwargs = inspect.signature(self._configure_cal_params).parameters
        cal_kwargs = {k: kwargs.pop(k) for k in cal_kwargs if k in kwargs}
        logger.info(f"Calibration model parameters: {cal_kwargs}")
        self._configure_cal_params(gw_dim, cal_dim, context_dim, **cal_kwargs)

        if isinstance(flow_class, str):
            try:
                FlowClass = getattr(zuko.flows, flow_class)
            except AttributeError:
                raise ValueError(f"Unknown flow_class: {flow_class}")
        else:
            FlowClass = flow_class

        self.flow_gw = FlowClass(
            features=gw_dim,
            context=context_dim or 0,
            hidden_features=list(hidden_features),
            **kwargs,
        )

    def _configure_cal_params(
        self, gw_dim: int, cal_dim: int, context_dim: Optional[int], **kwargs
    ) -> None:
        raise NotImplementedError

    def _cal_params(
        self, gw: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(
        self, context: Optional[torch.Tensor] = None
    ) -> "BaseGWCalFlow.JointDistribution":
        """
        Returns the joint distribution for GW and calibration parameters.

        Parameters
        ----------
        context : Optional[torch.Tensor], optional
            Context tensor, by default None.

        Returns
        -------
        JointDistribution
            Joint distribution object.
        """
        return self.JointDistribution(self, context)

    class JointDistribution:
        """
        Joint distribution for GW and calibration parameters.

        Parameters
        ----------
        parent : BaseGWCalFlow
            Parent flow object.
        context : Optional[torch.Tensor]
            Context tensor.
        """

        def __init__(
            self, parent: "BaseGWCalFlow", context: Optional[torch.Tensor]
        ) -> None:
            self.parent = parent
            self.context = context
            self.dist_gw = (
                parent.flow_gw(context)
                if parent.context_dim
                else parent.flow_gw()
            )

        def rsample(
            self, sample_shape: torch.Size = torch.Size()
        ) -> torch.Tensor:
            """
            Sample from the joint distribution.

            Parameters
            ----------
            sample_shape : torch.Size, optional
                Shape of samples, by default torch.Size().

            Returns
            -------
            torch.Tensor
                Sampled joint vector.
            """
            gw = self.dist_gw.sample(sample_shape)

            context = self.context
            if context is not None:
                while context.ndim < gw.ndim:
                    context = context.unsqueeze(0)
                context = context.expand(*gw.shape[:-1], context.shape[-1])

            loc, scale = self.parent._cal_params(gw, context)
            cal = zuko.distributions.DiagNormal(loc, scale, ndims=1).rsample()
            return self._assemble(gw, cal)

        def sample(
            self, sample_shape: torch.Size = torch.Size()
        ) -> torch.Tensor:
            """
            Sample from the joint distribution without gradients.

            Parameters
            ----------
            sample_shape : torch.Size, optional
                Shape of samples, by default torch.Size().

            Returns
            -------
            torch.Tensor
                Sampled joint vector.
            """
            with torch.no_grad():
                return self.rsample(sample_shape)

        def log_prob(self, x: torch.Tensor) -> torch.Tensor:
            """
            Compute log-probability of a sample.

            Parameters
            ----------
            x : torch.Tensor
                Input sample tensor.

            Returns
            -------
            torch.Tensor
                Log-probability value.
            """
            gw, cal = self._extract(x)
            logp_gw = self.dist_gw.log_prob(gw)
            loc, scale = self.parent._cal_params(gw, self.context)
            logp_cal = zuko.distributions.DiagNormal(
                loc, scale, ndims=1
            ).log_prob(cal)
            return logp_gw + logp_cal

        def rsample_and_log_prob(
            self, sample_shape: torch.Size = torch.Size()
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Sample from the joint distribution and compute log-probability.

            Parameters
            ----------
            sample_shape : torch.Size, optional
                Shape of samples, by default torch.Size().

            Returns
            -------
            Tuple[torch.Tensor, torch.Tensor]
                Sampled joint vector and its log-probability.
            """
            gw, logp_gw = self.dist_gw.rsample_and_log_prob(sample_shape)

            context = self.context
            if context is not None:
                while context.ndim < gw.ndim:
                    context = context.unsqueeze(0)
                context = context.expand(*gw.shape[:-1], context.shape[-1])

            loc, scale = self.parent._cal_params(gw, context)
            cal_dist = zuko.distributions.DiagNormal(loc, scale, ndims=1)
            cal = cal_dist.rsample()
            logp_cal = cal_dist.log_prob(cal)
            return self._assemble(gw, cal), logp_gw + logp_cal

        def _extract(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Extract GW and calibration parameters from input tensor.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor.

            Returns
            -------
            gw : torch.Tensor
                GW parameters.
            cal : torch.Tensor
                Calibration parameters.
            """
            raise NotImplementedError

        def _assemble(
            self, gw: torch.Tensor, cal: torch.Tensor
        ) -> torch.Tensor:
            """
            Assemble GW and calibration parameters into a single tensor.

            Parameters
            ----------
            gw : torch.Tensor
                GW parameters.
            cal : torch.Tensor
                Calibration parameters.

            Returns
            -------
            torch.Tensor
                Joint tensor.
            """
            raise NotImplementedError


class NNCalMixin:
    def _configure_cal_params(
        self,
        gw_dim: int,
        cal_dim: int,
        context_dim: Optional[int],
        cal_hidden_features: Sequence[int] = (128,),
    ) -> None:
        # Calibration network
        in_dim = gw_dim + (context_dim or 0)
        self.cal_net = zuko.nn.MLP(
            in_features=in_dim,
            out_features=cal_hidden_features[-1],
            hidden_features=cal_hidden_features,
            activation=nn.ELU,
        )
        self.loc_head = nn.Linear(cal_hidden_features[-1], cal_dim)
        self.log_scale_head = nn.Linear(cal_hidden_features[-1], cal_dim)
        nn.init.constant_(self.log_scale_head.bias, -2.0)

    def _cal_params(
        self, gw: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute location and scale for calibration parameters given GW and optional context.

        Parameters
        ----------
        gw : torch.Tensor
            GW parameter tensor.
        context : Optional[torch.Tensor], optional
            Context tensor, by default None.

        Returns
        -------
        loc : torch.Tensor
            Location tensor for calibration parameters.
        scale : torch.Tensor
            Scale tensor for calibration parameters.
        """
        if context is not None:
            while context.ndim < gw.ndim:
                context = context.unsqueeze(0)
            context = context.expand(*gw.shape[:-1], context.shape[-1])
            x = torch.cat([gw, context], dim=-1)
        else:
            x = gw

        h = self.cal_net(x)
        loc = self.loc_head(h)
        # log10 Scale between -2 and 2
        # scale = 10 ** (2.0 * torch.tanh(self.log_scale_head(h)))
        scale = torch.exp(self.log_scale_head(h)).clamp(min=1e-6, max=1e1)
        return loc, scale


class GaussianCalMixin:
    """Mixin for trainable Gaussian calibration parameters."""

    def _configure_cal_params(
        self, gw_dim: int, cal_dim: int, context_dim: Optional[int], **kwargs
    ) -> None:
        self.loc = nn.Parameter(torch.zeros(cal_dim))
        self.log_scale = nn.Parameter(torch.zeros(cal_dim))

    def _cal_params(
        self, gw: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return trainable location and scale for calibration parameters.

        Parameters
        ----------
        gw : torch.Tensor
            GW parameter tensor.
        context : Optional[torch.Tensor], optional
            Context tensor, by default None.

        Returns
        -------
        loc : torch.Tensor
            Location tensor for calibration parameters.
        scale : torch.Tensor
            Scale tensor for calibration parameters.
        """
        shape = gw.shape[:-1] + (self.loc.shape[0],)
        loc = self.loc.expand(shape)
        scale = (
            torch.exp(self.log_scale).clamp(min=1e-6, max=1e1).expand(shape)
        )
        return loc, scale


class ContiguousGWCalFlow(BaseGWCalFlow):
    """
    GWCalFlow with contiguous GW and calibration parameters.

    Must be used with a calibration mixin.

    Parameters
    ----------
    gw_dim : int
        Dimensionality of GW parameters.
    cal_dim : int
        Dimensionality of calibration parameters.
    context_dim : Optional[int], optional
        Dimensionality of context vector, by default None.
    hidden_features : Sequence[int], optional
        Hidden layer sizes for GW flow network, by default (128, 128).
    cal_hidden_features : Sequence[int], optional
        Hidden layer sizes for calibration network, by default (128,).
    """

    def __init__(
        self,
        gw_dim: int,
        cal_dim: int,
        context_dim: Optional[int] = None,
        hidden_features: Sequence[int] = (128, 128),
        **kwargs,
    ) -> None:
        super().__init__(
            gw_dim=gw_dim,
            cal_dim=cal_dim,
            context_dim=context_dim,
            hidden_features=hidden_features,
            **kwargs,
        )

    class JointDistribution(BaseGWCalFlow.JointDistribution):
        """
        Joint distribution for contiguous GW and calibration parameters.
        """

        def _extract(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Extract GW and calibration parameters from contiguous input tensor.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor.

            Returns
            -------
            gw : torch.Tensor
                GW parameters.
            cal : torch.Tensor
                Calibration parameters.
            """
            gw = x[..., : self.parent.gw_dim]
            cal = x[..., self.parent.gw_dim :]
            return gw, cal

        def _assemble(
            self, gw: torch.Tensor, cal: torch.Tensor
        ) -> torch.Tensor:
            """
            Assemble GW and calibration parameters into a contiguous tensor.

            Parameters
            ----------
            gw : torch.Tensor
                GW parameters.
            cal : torch.Tensor
                Calibration parameters.

            Returns
            -------
            torch.Tensor
                Joint tensor.
            """
            return torch.cat([gw, cal], dim=-1)


class IndexedGWCalFlow(BaseGWCalFlow):
    """
    GWCalFlow with indexed GW and calibration parameters.

    Must be used with a calibration mixin.

    Parameters
    ----------
    gw_idx : Sequence[int]
        Indices for GW parameters.
    cal_idx : Sequence[int]
        Indices for calibration parameters.
    context_dim : Optional[int], optional
        Dimensionality of context vector, by default None.
    hidden_features : Sequence[int], optional
        Hidden layer sizes for GW flow network, by default (128, 128).
    """

    def __init__(
        self,
        gw_idx: Sequence[int],
        cal_idx: Sequence[int],
        context_dim: Optional[int] = None,
        hidden_features: Sequence[int] = (128, 128),
        **kwargs,
    ) -> None:
        gw_dim = len(gw_idx)
        cal_dim = len(cal_idx)
        super().__init__(
            gw_dim=gw_dim,
            cal_dim=cal_dim,
            context_dim=context_dim,
            hidden_features=hidden_features,
            **kwargs,
        )
        self.gw_idx = torch.tensor(gw_idx, dtype=torch.long)
        self.cal_idx = torch.tensor(cal_idx, dtype=torch.long)

    class JointDistribution(BaseGWCalFlow.JointDistribution):
        """
        Joint distribution for indexed GW and calibration parameters.
        """

        def _extract(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Extract GW and calibration parameters from indexed input tensor.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor.

            Returns
            -------
            gw : torch.Tensor
                GW parameters.
            cal : torch.Tensor
                Calibration parameters.
            """
            gw = x[..., self.parent.gw_idx]
            cal = x[..., self.parent.cal_idx]
            return gw, cal

        def _assemble(
            self, gw: torch.Tensor, cal: torch.Tensor
        ) -> torch.Tensor:
            """
            Assemble GW and calibration parameters into a tensor using indices.

            Parameters
            ----------
            gw : torch.Tensor
                GW parameters.
            cal : torch.Tensor
                Calibration parameters.

            Returns
            -------
            torch.Tensor
                Joint tensor.
            """
            total_dim = gw.shape[-1] + cal.shape[-1]
            full_sample = torch.zeros(
                *gw.shape[:-1], total_dim, device=gw.device
            )
            full_sample[..., self.parent.gw_idx] = gw
            full_sample[..., self.parent.cal_idx] = cal
            return full_sample


class GWCalFlow:
    """
    Normalizing flow for joint distribution of GW and calibration parameters.

    Supports three initialization modes:
    1. By parameter indices: provide `gw_idx` and `cal_idx` lists.
    2. By parameter dimensions: provide `gw_dim` and `cal_dim` integers.
    3. By parameter names: provide `parameters` list and `calib_regex` string.
    """

    def __new__(
        cls,
        *,
        parameters: Optional[List[str]] = None,
        calib_regex: str = ".*calib.*",
        gw_dim: Optional[int] = None,
        cal_dim: Optional[int] = None,
        gw_idx: Optional[Sequence[int]] = None,
        cal_idx: Optional[Sequence[int]] = None,
        context_dim: Optional[int] = None,
        hidden_features: Sequence[int] = (128, 128),
        calibration_model: str = "nn",
        **kwargs,
    ) -> Union[ContiguousGWCalFlow, IndexedGWCalFlow]:
        """
        Factory method to construct a GWCalFlow using indices, dimensions, or parameter names.

        Parameters
        ----------
        parameters : Optional[List[str]], optional
            List of parameter names, by default None.
        calib_regex : str, optional
            Regex to identify calibration parameters, by default '.*calib.*'.
        gw_dim : Optional[int], optional
            Dimensionality of GW parameters, by default None.
        cal_dim : Optional[int], optional
            Dimensionality of calibration parameters, by default None.
        gw_idx : Optional[Sequence[int]], optional
            Indices for GW parameters, by default None.
        cal_idx : Optional[Sequence[int]], optional
            Indices for calibration parameters, by default None.
        context_dim : Optional[int], optional
            Dimensionality of context vector, by default None.
        hidden_features : Sequence[int], optional
            Hidden layer sizes for GW flow network, by default (128, 128).

        Returns
        -------
        ContiguousGWCalFlow or IndexedGWCalFlow
            GWCalFlow instance.
        """
        import re

        if calibration_model == "nn":
            CalMixin = NNCalMixin
        elif calibration_model == "gaussian":
            CalMixin = GaussianCalMixin
        else:
            raise ValueError(f"Unknown calibration_type: {calibration_model}")

        if parameters is not None:
            cal_idx = [
                i
                for i, p in enumerate(parameters)
                if re.search(calib_regex, p)
            ]
            gw_idx = [i for i in range(len(parameters)) if i not in cal_idx]
            logger.info(
                f"Identified {len(gw_idx)} GW parameters and {len(cal_idx)} calibration parameters."
            )

            class GWCalFlowWithCalMixin(CalMixin, IndexedGWCalFlow):
                pass

            return GWCalFlowWithCalMixin(
                gw_idx=gw_idx,
                cal_idx=cal_idx,
                context_dim=context_dim,
                hidden_features=hidden_features,
                **kwargs,
            )
        elif gw_dim is not None and cal_dim is not None:

            class GWCalFlowWithCalMixin(CalMixin, ContiguousGWCalFlow):
                pass

            return GWCalFlowWithCalMixin(
                gw_dim=gw_dim,
                cal_dim=cal_dim,
                context_dim=context_dim,
                hidden_features=hidden_features,
                **kwargs,
            )
        elif gw_idx is not None and cal_idx is not None:

            class GWCalFlowWithCalMixin(CalMixin, IndexedGWCalFlow):
                pass

            return GWCalFlowWithCalMixin(
                gw_idx=gw_idx,
                cal_idx=cal_idx,
                context_dim=context_dim,
                hidden_features=hidden_features,
                **kwargs,
            )
        else:
            raise ValueError(
                "Must provide either (parameters and calib_regex) or (gw_dim and cal_dim) or (gw_idx and cal_idx)."
            )

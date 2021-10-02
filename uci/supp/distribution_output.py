
from abc import ABC, abstractclassmethod
import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pts.distributions import (
    ZeroInflatedPoisson,
    ZeroInflatedNegativeBinomial,
    PiecewiseLinear,
    TransformedPiecewiseLinear,
    ImplicitQuantile,
    TransformedImplicitQuantile,
)
from gluonts.core.component import validated
from gluonts.torch.modules.distribution_output import (
    DistributionOutput,
    LambdaLayer,
    PtArgProj,
)

class PiecewiseLinearOutput(DistributionOutput):
    distr_cls: type = PiecewiseLinear

    @validated()
    def __init__(self, num_pieces: int) -> None:
        super().__init__(self)
        assert (
            isinstance(num_pieces, int) and num_pieces > 1
        ), "num_pieces should be an integer larger than 1"

        self.num_pieces = num_pieces
        self.args_dim = {"gamma": 1, "slopes": num_pieces, "knot_spacings": num_pieces}

    @classmethod
    def domain_map(cls, gamma, slopes, knot_spacings):
        # slopes of the pieces are non-negative
        slopes_proj = F.softplus(slopes) + 1e-4

        # the spacing between the knots should be in [0, 1] and sum to 1
        knot_spacings_proj = torch.softmax(knot_spacings, dim=-1)

        return gamma.squeeze(axis=-1), slopes_proj, knot_spacings_proj

    def distribution(
        self,
        distr_args,
        scale: Optional[torch.Tensor] = None,
    ) -> PiecewiseLinear:
        if scale is None:
            return self.distr_cls(*distr_args)
        else:
            distr = self.distr_cls(*distr_args)
            return TransformedPiecewiseLinear(
                distr, [AffineTransform(loc=0, scale=scale)]
            )

    @property
    def event_shape(self) -> Tuple:
        return ()

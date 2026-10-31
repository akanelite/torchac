# Derived from torchac (https://github.com/fab-jul/torchac), Copyright 2020 ETH Zurich.
# Modified 2026: rewritten with type hints and a new public API.
# Licensed under the GNU General Public License v3.0; see LICENSE.

from typing import Sequence

import numpy as np
import torch

from . import _backend


__all__ = [
    "encode",
    "decode",
    "encode_int16_normalized_cdf",
    "decode_int16_normalized_cdf",
]


# Types

PRECISION = 16

Tensor = torch.Tensor


# Helper Functions

def _check_and_reshape_inputs(cdf: Tensor, sym: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
    if cdf.is_cuda:
        raise ValueError('CDF must be on CPU')
    if sym is not None and sym.is_cuda:
        raise ValueError('Symbols must be on CPU')
    if sym is not None and sym.dtype != torch.int16:
        raise ValueError('Symbols must be int16!')
    if sym is not None:
        if len(cdf.shape) != len(sym.shape) + 1 or cdf.shape[:-1] != sym.shape:
            raise ValueError(f'Invalid shapes of cdf={cdf.shape}, sym={sym.shape}! '
                             f'The first m elements of cdf.shape must be equal to '
                             f'sym.shape, and cdf should only have one more dimension.')
    cdf = cdf.reshape(-1, cdf.shape[-1])
    if sym is None:
        return cdf
    sym = sym.reshape(-1)
    return cdf, sym


def _convert_to_int_and_normalize(cdf_float: Tensor, needs_normalization: bool) -> Tensor:
    factor = torch.tensor(2, dtype=torch.float32, device=cdf_float.device).pow_(PRECISION)
    new_max_value = factor
    if needs_normalization:
        new_max_value = new_max_value - (cdf_float.shape[-1] - 1)
    cdf_float = cdf_float.mul(new_max_value)
    cdf_float = cdf_float.round()
    cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
    if needs_normalization:
        r = torch.arange(cdf_float.shape[-1], dtype=torch.int16, device=cdf.device)
        cdf.add_(r)
    return cdf


def _reshape_output(cdf_shape: Sequence[int], sym: Tensor) -> Tensor:
    spatial_dimensions = cdf_shape[:-1]
    if len(sym) != np.prod(spatial_dimensions):
        raise ValueError()
    return sym.reshape(*spatial_dimensions)


# Functional

def encode(sym: Tensor, cdf: Tensor, needs_normalization: bool = False, check_input_bounds: bool = False) -> bytes:  # noqa: B950
    if check_input_bounds:
        if cdf.min() < 0:
            raise ValueError(f'cdf.min() == {cdf.min()}, should be >=0')
        if cdf.max() > 1:
            raise ValueError(f'cdf.max() == {cdf.max()}, should be <=1')
        if sym.max() >= cdf.shape[-1] - 1:
            raise ValueError
    cdf = _convert_to_int_and_normalize(cdf, needs_normalization)
    return encode_int16_normalized_cdf(sym, cdf)


def encode_int16_normalized_cdf(sym: Tensor, cdf: Tensor) -> bytes:
    cdf, sym = _check_and_reshape_inputs(cdf, sym)
    return _backend.encode_cdf(cdf, sym)


def decode(byte_stream: bytes, cdf: Tensor, needs_normalization: bool = False) -> Tensor:
    cdf = _convert_to_int_and_normalize(cdf, needs_normalization)
    return decode_int16_normalized_cdf(byte_stream, cdf)


def decode_int16_normalized_cdf(byte_stream: bytes, cdf: Tensor) -> Tensor:
    cdf_reshaped = _check_and_reshape_inputs(cdf)
    sym = _backend.decode_cdf(cdf_reshaped, byte_stream)
    return _reshape_output(cdf.shape, sym)

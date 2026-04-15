#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytorch physics functions
"""
import torch
import torch.nn as nn
import torch.nn.parameter as Parameter
import torch.nn.functional as F
from torch import Tensor

def reitab(t: torch.Tensor) -> torch.Tensor:
    """
    Interpolate effective radius from temperature using tabulated values.
    
    Args:
        t: Temperature tensor of shape (N,) where N = ncol * pver (or any flat shape)
    
    Returns:
        re: Effective radius tensor, same shape as t
    """
    retab = torch.tensor([
        0.05,      0.05,      0.05,      0.05,      0.05,      0.05,
        0.055,     0.06,      0.07,      0.08,      0.09,      0.1,
        0.2,       0.3,       0.40,      0.50,      0.60,      0.70,
        0.8,       0.9,       1.0,       1.1,       1.2,       1.3,
        1.4,       1.5,       1.6,       1.8,       2.0,       2.2,
        2.4,       2.6,       2.8,       3.0,       3.2,       3.5,
        3.8,       4.1,       4.4,       4.7,       5.0,       5.3,
        5.6,
        5.92779,   6.26422,   6.61973,   6.99539,   7.39234,
        7.81177,   8.25496,   8.72323,   9.21800,   9.74075,   10.2930,
        10.8765,   11.4929,   12.1440,   12.8317,   13.5581,   14.2319,
        15.0351,   15.8799,   16.7674,   17.6986,   18.6744,   19.6955,
        20.7623,   21.8757,   23.0364,   24.2452,   25.5034,   26.8125,
        27.7895,   28.6450,   29.4167,   30.1088,   30.7306,   31.2943,
        31.8151,   32.3077,   32.7870,   33.2657,   33.7540,   34.2601,
        34.7892,   35.3442,   35.9255,   36.5316,   37.1602,   37.8078,
        38.4720,   39.1508,   39.8442,   40.5552,   41.2912,   42.0635,
        42.8876,   43.7863,   44.7853,   45.9170,   47.2165,   48.7221,
        50.4710,   52.4980,   54.8315,   57.4898,   60.4785,   63.7898,
        65.5604,   71.2885,   75.4113,   79.7368,   84.2351,   88.8833,
        93.6658,   98.5739,   103.603,   108.752,   114.025,   119.424,
        124.954,   130.630,   136.457,   142.446,   148.608,   154.956,
        161.503,   168.262,   175.248,   182.473,   189.952,   197.699,
        205.728,   214.055,   222.694,   231.661,   240.971,   250.639,
    ], dtype=t.dtype, device=t.device)

    min_retab = 136.0
    len_retab = len(retab)  # 138

    index = (t - min_retab).to(torch.int32)
    index = index.clamp(1, len_retab - 2)  # clamp to [1, 136] for safe index+1 access

    corr = t - t.floor()

    re = retab[index] * (1.0 - corr) + retab[index + 1] * corr

    return re

def reltab(
        t: torch.Tensor,          # (N,) flattened from (ncol, pver)
        landfrac: torch.Tensor,   # (N,) — repeated across levels by caller
        icefrac: torch.Tensor,    # (N,) — repeated across levels by caller
        snowh: torch.Tensor,      # (N,) — repeated across levels by caller
    ) -> torch.Tensor:
        """
        Compute liquid cloud droplet effective radius.

        Args:
            t:        Temperature, shape (N,) where N = ncol * pver
            landfrac: Land fraction, shape (N,) — ncol values repeated pver times
            icefrac:  Ice fraction,  shape (N,) — ncol values repeated pver times
            snowh:    Snow depth (m, water equivalent), shape (N,) — same tiling

        Returns:
            rel: Liquid effective drop size in microns, shape (N,)
        """
        rliqocean = 14.0
        rliqice   = 14.0
        rliqland  =  8.0

        # Temperature-dependent baseline: continental air, ramped toward ocean value when cold
        rel = rliqland + (rliqocean - rliqland) * ((273.15 - t) * 0.05).clamp(0.0, 1.0)

        # Modify for snow depth over land
        rel = rel + (rliqocean - rel) * (snowh * 10.0).clamp(0.0, 1.0)

        # Ramp from land (polluted) to ocean (clean)
        rel = rel + (rliqocean - rel) * (1.0 - landfrac).clamp(0.0, 1.0)

        # Ramp toward sea-ice value in presence of ice
        rel = rel + (rliqice - rel) * icefrac.clamp(0.0, 1.0)

        return rel

def slingo_liq_cloud_optics_sw(rel):
    # Adapted from https://github.com/NVlabs/E3SM/blob/main/components/eam/src/physics/rrtmgp/slingo.F90#L32
    coeffs1 = torch.tensor([2.817e-02, 2.682e-02, 2.264e-02, 1.281e-02], dtype=rel.dtype, device=rel.device)  # A: extinction OD
    coeffs2 = torch.tensor([1.305,     1.346,     1.454,     1.641    ], dtype=rel.dtype, device=rel.device)  # B: extinction OD
    coeffs3 = torch.tensor([-5.62e-08, -6.94e-06, 4.64e-04,  0.201    ], dtype=rel.dtype, device=rel.device)  # C: single scat albedo
    coeffs4 = torch.tensor([1.63e-07,  2.35e-05,  1.24e-03,  7.56e-03 ], dtype=rel.dtype, device=rel.device)  # D: single scat albedo
    coeffs5 = torch.tensor([0.829,     0.794,     0.754,     0.826    ], dtype=rel.dtype, device=rel.device)  # E: asymmetry parameter
    coeffs6 = torch.tensor([2.482e-03, 4.226e-03, 6.560e-03, 4.353e-03], dtype=rel.dtype, device=rel.device)  # F: asymmetry parameter
    re_um   = rel.clamp(4.2, 16.6)

    k       = (coeffs1 + coeffs2 / re_um)
    ssa     = (1.0 - coeffs3 - re_um * coeffs4)
    g       = coeffs5 + re_um * coeffs6
    return k, ssa, g 

def ec_ice_optics_sw(rei):
    # Adapted from https://github.com/NVlabs/E3SM/blob/main/components/eam/src/physics/rrtmgp/ebert_curry.F90#L30
    coeffs1 = torch.tensor([3.448e-03, 3.448e-03, 3.448e-03, 3.448e-03], dtype=rei.dtype, device=rei.device)  # a: extinction OD
    coeffs2 = torch.tensor([2.431,     2.431,     2.431,     2.431    ], dtype=rei.dtype, device=rei.device)  # b: extinction OD
    coeffs3 = torch.tensor([1.00e-05,  1.10e-04,  1.861e-02, 0.46658  ], dtype=rei.dtype, device=rei.device)  # c: single scat albedo
    coeffs4 = torch.tensor([0.0,       1.405e-05, 8.328e-04, 2.05e-05 ], dtype=rei.dtype, device=rei.device)  # d: single scat albedo
    coeffs5 = torch.tensor([0.7661,    0.7730,    0.794,     0.9595   ], dtype=rei.dtype, device=rei.device)  # e: asymmetry parameter
    coeffs6 = torch.tensor([5.851e-04, 5.665e-04, 7.267e-04, 1.076e-04], dtype=rei.dtype, device=rei.device)  # f: asymmetry parameter

    re_um   = rei.clamp(13.0, 130.0)
    k       = (coeffs1 + coeffs2 / re_um)
    ssa     = (1.0 - coeffs3 - re_um * coeffs4)
    g       = coeffs5 + re_um * coeffs6
    return k, ssa, g 
    
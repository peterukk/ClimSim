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
from typing import Sequence, Optional, List

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

def slingo_liq_cloud_optics_sw(rel:torch.Tensor, ng:int=4):
    # Input: rel (cloud liquid effective radius), ng (number of g-points/bands)
    # Here we compute cloud optical properties using the Slingo scheme, which uses 4 different bands 
    # We also inline the mapping of the cloud optical properties in these 4 bands to our g-point discretization
    # If our gas optics model equals RRTMGP, then this function should give similar results as original E3SM code (links below)
    # Note the McICA sampling is done outside - here we compute "normalized" cloud optical properties which are later multiplied 
    # with cloud water paths which are sampled using McICA
    # Adapted from https://github.com/NVlabs/E3SM/blob/main/components/eam/src/physics/rrtmgp/slingo.F90#L32
    # https://github.com/NVlabs/E3SM/blob/c92b443306bd8e7c3796ccc9966d43190da992d5/components/eam/src/physics/rrtmgp/radiation.F90#L1290
    # Slingo look-up-table coefficients in vector form (length 4 equals number of bands)
    coeffs1 = torch.tensor([2.817e-02, 2.682e-02, 2.264e-02, 1.281e-02], dtype=rel.dtype, device=rel.device)  # A: extinction OD
    coeffs2 = torch.tensor([1.305,     1.346,     1.454,     1.641    ], dtype=rel.dtype, device=rel.device)  # B: extinction OD
    coeffs3 = torch.tensor([-5.62e-08, -6.94e-06, 4.64e-04,  0.201    ], dtype=rel.dtype, device=rel.device)  # C: single scat albedo
    coeffs4 = torch.tensor([1.63e-07,  2.35e-05,  1.24e-03,  7.56e-03 ], dtype=rel.dtype, device=rel.device)  # D: single scat albedo
    coeffs5 = torch.tensor([0.829,     0.794,     0.754,     0.826    ], dtype=rel.dtype, device=rel.device)  # E: asymmetry parameter
    coeffs6 = torch.tensor([2.482e-03, 4.226e-03, 6.560e-03, 4.353e-03], dtype=rel.dtype, device=rel.device)  # F: asymmetry parameter

    re_um   = rel.clamp(4.2, 16.0)
    # RRTMGP bands and g-points:
    # bnd_limits_gpt
    # 1,10     | 11,18    | 19,29    | 30,37   | 38,46     | 47,56     | 57,67     | 68,71      | 72,80      |  81,89    | 90, 96    | 97, 102   | 103, 109 | 110, 112 
    # bnd_limits_wavenumber
    # 820,2680 | 2680,3250 | 3250,4k | 4k,4650 | 4650,5150 | 5150,6150 | 6150,7700 | 7700,8050  | 8050,12850 | 12850,16k | 16k,22650 | 22650,29k | 29k,38k  | 38k,50k 
    # in micrometers 
    # 12.2,3.73| 3.73,3.08 | 3.08,2.5| 2.5,2.15| 2.15,1.94 | 1.94,1.63 | 1.63,1.3  | 1.3,1.24   | 1.24,0.78  | 0.78,0.62 | 0.62,0.44 | 0.44,0.34 | 0.34,0.26| 0.26  0.2 ]
    # band 1        2         3           4           5         6             7         8              9           10         11          12         13         14
    if ng == 4:
        # if ng=4 (default argument), we just output the original bands (but reversed as RRTMGP uses different order as coeffs1, coeffs2..)
        # We can then map to a custom gas optics scheme by a learned mapping
        k       = (coeffs1 + coeffs2 / re_um)
        ssa     = (1.0 - coeffs3 - re_um * coeffs4).clamp(max=0.999999)
        g       = coeffs5 + re_um * coeffs6
    else:
        # if ng is not 4: 
        #  For ng=112, we map to RRTMGP's g-points using knowledge of in which bands its 112 g-points are (should be equivalent to E3SM code)
        #  if ng is anything else, we use a similar band allocation as RRTMGP (assuming N g-points are divided into the same bands as RRTMGP, just with fewer g-points in each)
        x = torch.stack([coeffs1, coeffs2, coeffs3, coeffs4, coeffs5, coeffs6])  
        y = torch.empty(6, ng, dtype=rel.dtype, device=rel.device)

        i_lim_b4 = int(round((29/112)*ng))   # g-points 0..i_lim_b4      → Slingo band 4 (coeffs index 3)
        i_lim_b3 = int(round((71/112)*ng))   # g-points i_lim_b4..i_lim_b3 → Slingo band 3 (coeffs index 2)
        i_lim_b2 = int(round((80/112)*ng))   # g-points i_lim_b3..i_lim_b2 → Slingo band 2 (coeffs index 1)
        #                                      g-points i_lim_b2..        → Slingo band 1 (coeffs index 0)

        y[:, 0:i_lim_b4]         = x[:, 3:4]   # Slingo band 4
        y[:, i_lim_b4:i_lim_b3]  = x[:, 2:3]   # Slingo band 3
        y[:, i_lim_b3:i_lim_b2]  = x[:, 1:2]   # Slingo band 2
        y[:, i_lim_b2:]          = x[:, 0:1]   # Slingo band 1

        # i_lim1 = int(round((29/112)*ng))
        # i_lim2 = int(round((37/112)*ng))
        # i_lim3 = int(round((67/112)*ng))
        # i_lim4 = int(round((71/112)*ng))
        # i_lim5 = int(round((80/112)*ng))
        # i_lim6 = int(round((89/112)*ng))
        # y[:, 0:i_lim1]      = x[:, 3:4]
        # y[:, i_lim1:i_lim2] = 0.5 * (x[:, 2:3] + x[:, 3:4])
        # y[:, i_lim2:i_lim3] = x[:, 2:3]
        # y[:, i_lim3:i_lim4] = 0.5 * (x[:, 1:2] + x[:, 2:3])
        # y[:, i_lim4:i_lim5] = x[:, 1:2]
        # y[:, i_lim5:i_lim6] = 0.5 * (x[:, 0:1] + x[:, 1:2])
        # y[:, i_lim6:]       = x[:, 0:1]
        
        # Original Fortran code for Slingo cloud optics computations in 4 specific bands commented out below:
        # subroutine slingo_liq_optics_sw(ncol, nlev, cldn, cliqwp, rel, liq_tau, liq_tau_w, liq_tau_w_g, liq_tau_w_f)

        # integer, intent(in) :: ncol, nlev

        # ! Inputs have dimension ncol,nlev
        # real(r8), intent(in), dimension(:,:) :: rel
        # real(r8), intent(in), dimension(:,:) :: cldn
        # real(r8), intent(in), dimension(:,:) :: cliqwp 

        # ! Outputs have dimension nbnd,ncol,nlev
        # real(r8),intent(out) :: liq_tau    (:,:,:) ! extinction optical depth
        # real(r8),intent(out) :: liq_tau_w  (:,:,:) ! single scattering albedo * tau
        # real(r8),intent(out) :: liq_tau_w_g(:,:,:) ! assymetry parameter * tau * w
        # real(r8),intent(out) :: liq_tau_w_f(:,:,:) ! forward scattered fraction * tau * w

        # real(r8), dimension(nswbands)     :: wavmin
        # real(r8), dimension(nswbands)     :: wavmax

        # ...

        # call get_sw_spectral_boundaries(wavmin,wavmax,'microns')
        # do ns = 1, nswbands
        #     ! Set index for cloud particle properties based on the wavelength,
        #     ! according to A. Slingo (1989) equations 1-3:
        #     ! Use index 1 (0.25 to 0.69 micrometers) for visible
        #     ! Use index 2 (0.69 - 1.19 micrometers) for near-infrared
        #     ! Use index 3 (1.19 to 2.38 micrometers) for near-infrared
        #     ! Use index 4 (2.38 to 4.00 micrometers) for near-infrared
        #     if(wavmax(ns) <= 0.7_r8) then
        #     indxsl = 1
        #     else if(wavmax(ns) <= 1.25_r8) then
        #     indxsl = 2
        #     else if(wavmax(ns) <= 2.38_r8) then
        #     indxsl = 3
        #     else if(wavmax(ns) > 2.38_r8) then
        #     indxsl = 4
        #     end if
        #     ! Set cloud extinction optical depth, single scatter albedo,
        #     ! asymmetry parameter, and forward scattered fraction:
        #     abarli = abarl(indxsl)
        #     bbarli = bbarl(indxsl)
        #     cbarli = cbarl(indxsl)
        #     dbarli = dbarl(indxsl)
        #     ebarli = ebarl(indxsl)
        #     fbarli = fbarl(indxsl)

        #     do k=1,nlev
        #         do i=1,ncol

        #         ! note that optical properties for liquid valid only
        #         ! in range of 4.2 > rel > 16 micron (Slingo 89)
        #         if (cldn(i,k) >= cldmin .and. cldn(i,k) >= cldeps) then
        #             tmp1l = abarli + bbarli/min(max(rel_min,rel(i,k)),rel_max)
        #             liq_tau(ns,i,k) = 1000._r8*cliqwp(i,k)*tmp1l
        #         else
        #             liq_tau(ns,i,k) = 0.0_r8
        #         endif

        #         tmp2l = 1._r8 - cbarli - dbarli*min(max(rel_min,rel(i,k)),rel_max)
        #         tmp3l = fbarli*min(max(rel_min,rel(i,k)),rel_max)
        #     ...

        # ! Fortran code for reordering from RRTMG order to RRTMGP
        # real(r8),parameter :: wavenum_low(nbndsw) = & ! in cm^-1
        # (/2600._r8, 3250._r8, 4000._r8, 4650._r8, 5150._r8, 6150._r8, 7700._r8, 8050._r8,12850._r8,16000._r8,22650._r8,29000._r8,38000._r8,  820._r8/)
        # real(r8),parameter :: wavenum_high(nbndsw) = & ! in cm^-1
        # (/3250._r8, 4000._r8, 4650._r8, 5150._r8, 6150._r8, 7700._r8, 8050._r8, 12850._r8,16000._r8,22650._r8,29000._r8,38000._r8,50000._r8, 2600._r8/)
        # ! Mapping from old RRTMG sw bands to new band ordering in RRTMGP
        # integer, parameter, dimension(14), public :: rrtmg_to_rrtmgp_swbands = (/ 14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 /)
        
        # ! We need to reorder the spectral bounds since we store them in RRTMG order in radconstants!
        # lower_bounds = reordered(wavenum_low, rrtmg_to_rrtmgp_swbands)
        # upper_bounds = reordered(wavenum_high, rrtmg_to_rrtmgp_swbands)

        # ! Utility function to reorder an array given a new indexing
        # function reordered(array_in, new_indexing) result(array_out)
        #     ...
        #     ! Reorder array based on input index mapping, which maps old indices to new
        #     do ii = 1,size(new_indexing)
        #         array_out(ii) = array_in(new_indexing(ii))
        #     end do
        # end function reordered
        
        # do icol = 1,size(cld_tau_bnd_sw,1)
        #     do ilay = 1,size(cld_tau_bnd_sw,2)
        #        cld_tau_bnd_sw(icol,ilay,:) = reordered(cld_tau_bnd_sw(icol,ilay,:), rrtmg_to_rrtmgp_swbands)
        #        cld_ssa_bnd_sw(icol,ilay,:) = reordered(cld_ssa_bnd_sw(icol,ilay,:), rrtmg_to_rrtmgp_swbands)
        #        cld_asm_bnd_sw(icol,ilay,:) = reordered(cld_asm_bnd_sw(icol,ilay,:), rrtmg_to_rrtmgp_swbands)
        #     end do
        #  end do
        #  ! And now do the MCICA sampling to get cloud optical properties by gpoint/cloud state
        #  call get_gpoint_bands_sw(gpoint_bands_sw)
        #  call sample_cloud_optics_sw( &
        #     ncol, pver, nswgpts, gpoint_bands_sw, &
        #     state%pmid, cld, cldfsnow, &
        #     cld_tau_bnd_sw, cld_ssa_bnd_sw, cld_asm_bnd_sw, &
        #     cld_tau_gpt_sw, cld_ssa_gpt_sw, cld_asm_gpt_sw &
        #  )          
        k       = (y[0] + y[1] / re_um)
        ssa     = (1.0 - y[2] - re_um * y[3]).clamp(max=0.999999)
        g       = y[4] + re_um * y[5]   
    return k, ssa, g 

def ec_ice_optics_sw(rei:torch.Tensor, ng:int=4):
    # Adapted from https://github.com/NVlabs/E3SM/blob/main/components/eam/src/physics/rrtmgp/ebert_curry.F90#L30
    coeffs1 = torch.tensor([3.448e-03, 3.448e-03, 3.448e-03, 3.448e-03], dtype=rei.dtype, device=rei.device)  # a: extinction OD
    coeffs2 = torch.tensor([2.431,     2.431,     2.431,     2.431    ], dtype=rei.dtype, device=rei.device)  # b: extinction OD
    coeffs3 = torch.tensor([1.00e-05,  1.10e-04,  1.861e-02, 0.46658  ], dtype=rei.dtype, device=rei.device)  # c: single scat albedo
    coeffs4 = torch.tensor([0.0,       1.405e-05, 8.328e-04, 2.05e-05 ], dtype=rei.dtype, device=rei.device)  # d: single scat albedo
    coeffs5 = torch.tensor([0.7661,    0.7730,    0.794,     0.9595   ], dtype=rei.dtype, device=rei.device)  # e: asymmetry parameter
    coeffs6 = torch.tensor([5.851e-04, 5.665e-04, 7.267e-04, 1.076e-04], dtype=rei.dtype, device=rei.device)  # f: asymmetry parameter

    re_um   = rei.clamp(13.0, 130.0)

    if ng != 4:
        x = torch.stack([coeffs1, coeffs2, coeffs3, coeffs4, coeffs5, coeffs6])  
        y = torch.empty(6, ng, dtype=rei.dtype, device=rei.device)

        i_lim_b4 = int(round((29/112)*ng))   # g-points 0..i_lim_b4      → Slingo band 4 (coeffs index 3)
        i_lim_b3 = int(round((71/112)*ng))   # g-points i_lim_b4..i_lim_b3 → Slingo band 3 (coeffs index 2)
        i_lim_b2 = int(round((80/112)*ng))   # g-points i_lim_b3..i_lim_b2 → Slingo band 2 (coeffs index 1)
        #                                      g-points i_lim_b2..        → Slingo band 1 (coeffs index 0)
        y[:, 0:i_lim_b4]         = x[:, 3:4]   # Slingo band 4
        y[:, i_lim_b4:i_lim_b3]  = x[:, 2:3]   # Slingo band 3
        y[:, i_lim_b3:i_lim_b2]  = x[:, 1:2]   # Slingo band 2
        y[:, i_lim_b2:]          = x[:, 0:1]   # Slingo band 1

        k       = (y[0] + y[1] / re_um)
        ssa     = (1.0 - y[2] - re_um * y[3]).clamp(max=0.999999)
        g       = y[4] + re_um * y[5]   
    else:
         
        k       = (coeffs1 + coeffs2 / re_um)
        ssa     = (1.0 - coeffs3 - re_um * coeffs4).clamp(max=0.999999)
        g       = coeffs5 + re_um * coeffs6
    return k, ssa, g 




def e3sm_cloud_optics_sw(
    re: torch.Tensor,
    wavenum_bounds: List[int],
    band_bounds: List[int],
    type: str,
):
    
    # Input: rel (cloud liquid effective radius), ng (number of g-points/bands)
    # Here we compute cloud optical properties using the Slingo scheme, which uses 4 different bands 
    # We also inline the mapping of the cloud optical properties in these 4 bands to our g-point discretization
    # If our gas optics model equals RRTMGP, then this function should give similar results as original E3SM code (links below)
    # Note the McICA sampling is done outside - here we compute "normalized" cloud optical properties which are later multiplied 
    # with cloud water paths which are sampled using McICA
    # Adapted from https://github.com/NVlabs/E3SM/blob/main/components/eam/src/physics/rrtmgp/slingo.F90#L32
    # https://github.com/NVlabs/E3SM/blob/c92b443306bd8e7c3796ccc9966d43190da992d5/components/eam/src/physics/rrtmgp/radiation.F90#L1290
    # Slingo look-up-table coefficients in vector form (length 4 equals number of bands)
    if type=="liquid":
        coeffs1 = torch.tensor([2.817e-02, 2.682e-02, 2.264e-02, 1.281e-02], dtype=re.dtype, device=re.device)  # A: extinction OD
        coeffs2 = torch.tensor([1.305,     1.346,     1.454,     1.641    ], dtype=re.dtype, device=re.device)  # B: extinction OD
        coeffs3 = torch.tensor([-5.62e-08, -6.94e-06, 4.64e-04,  0.201    ], dtype=re.dtype, device=re.device)  # C: single scat albedo
        coeffs4 = torch.tensor([1.63e-07,  2.35e-05,  1.24e-03,  7.56e-03 ], dtype=re.dtype, device=re.device)  # D: single scat albedo
        coeffs5 = torch.tensor([0.829,     0.794,     0.754,     0.826    ], dtype=re.dtype, device=re.device)  # E: asymmetry parameter
        coeffs6 = torch.tensor([2.482e-03, 4.226e-03, 6.560e-03, 4.353e-03], dtype=re.dtype, device=re.device)  # F: asymmetry parameter
        # print("liq")
        re_um   = re.clamp(4.2, 16.0)
    elif type=="ice":
        coeffs1 = torch.tensor([3.448e-03, 3.448e-03, 3.448e-03, 3.448e-03], dtype=re.dtype, device=re.device)  # a: extinction OD
        coeffs2 = torch.tensor([2.431,     2.431,     2.431,     2.431    ], dtype=re.dtype, device=re.device)  # b: extinction OD
        coeffs3 = torch.tensor([1.00e-05,  1.10e-04,  1.861e-02, 0.46658  ], dtype=re.dtype, device=re.device)  # c: single scat albedo
        coeffs4 = torch.tensor([0.0,       1.405e-05, 8.328e-04, 2.05e-05 ], dtype=re.dtype, device=re.device)  # d: single scat albedo
        coeffs5 = torch.tensor([0.7661,    0.7730,    0.794,     0.9595   ], dtype=re.dtype, device=re.device)  # e: asymmetry parameter
        coeffs6 = torch.tensor([5.851e-04, 5.665e-04, 7.267e-04, 1.076e-04], dtype=re.dtype, device=re.device)  # f: asymmetry parameter
        re_um   = re.clamp(13.0, 130.0)
        # print("ice")
    else:
        raise NotImplementedError("invalid type arg to e3sm_cloud_optics_sw")

    # RRTMGP bands and g-points:
    # bnd_limits_gpt
    # 1,10     | 11,18    | 19,29    | 30,37   | 38,46     | 47,56     | 57,67     | 68,71      | 72,80      |  81,89    | 90, 96    | 97, 102   | 103, 109 | 110, 112 
    # bnd_limits_wavenumber
    # 820,2680 | 2680,3250 | 3250,4k | 4k,4650 | 4650,5150 | 5150,6150 | 6150,7700 | 7700,8050  | 8050,12850 | 12850,16k | 16k,22650 | 22650,29k | 29k,38k  | 38k,50k 
    # in micrometers 
    # 12.2,3.73| 3.73,3.08 | 3.08,2.5| 2.5,2.15| 2.15,1.94 | 1.94,1.63 | 1.63,1.3  | 1.3,1.24   | 1.24,0.78  | 0.78,0.62 | 0.62,0.44 | 0.44,0.34 | 0.34,0.26| 0.26  0.2 ]
    # band 1        2         3           4           5         6             7         8              9           10         11          12         13         14

    n_bands = len(wavenum_bounds) - 1
    # gpt_counts = band_bounds[1:] - band_bounds[0:]
    bb = band_bounds
    # ng = int(sum(gpt_counts))

    # Convert increasing wavenumber boundaries to gas-band lower edges.
    # For each gas band [nu_low, nu_high], maximum wavelength is
    # lambda_max = 10000 / nu_low, with nu in cm^-1 and lambda in micron.
    nu_low = torch.tensor(
        list(wavenum_bounds[:-1]),
        dtype=re.dtype,
        device=re.device,
    )
    wavmax_um = 10000.0 / nu_low

    slingo_idx = torch.bucketize(
        wavmax_um,
        torch.tensor([0.7, 1.25, 2.38], dtype=re.dtype, device=re.device),
        right=True,
    )


    # Expand gas-band mapping to g-points if requested.
    # This preserves the user's gas-optics ordering, assumed to be
    # increasing wavenumber / decreasing wavelength.
    repeats = torch.tensor(
        [bb[i + 1] - bb[i] for i in range(len(bb) - 1)],
        dtype=torch.long,
        device=re.device,
    )

    band_to_gpt = torch.repeat_interleave(slingo_idx, repeats)

    x = torch.stack([coeffs1, coeffs2, coeffs3, coeffs4, coeffs5, coeffs6])
    y = x[:, band_to_gpt]

    k = y[0] + y[1] / re_um
    ssa = (1.0 - y[2] - re_um * y[3]).clamp(max=0.999999)
    g = y[4] + re_um * y[5]

    return k, ssa, g     


# class slingo_style_general_liq_cloud_optics_sw:
#     def __init__(self, ng:int=16):
#         super(slingo_style_general_liq_cloud_optics_sw, self).__init__()
#         self.ng = ng
#         self.coeffs1 = nn.Parameter(torch.Tensor(ng))
#         self.coeffs2 = nn.Parameter(torch.Tensor(ng))
#         self.coeffs4 = nn.Parameter(torch.Tensor(ng))
#         self.coeffs5 = nn.Parameter(torch.Tensor(ng))
#         self.coeffs6 = nn.Parameter(torch.Tensor(ng))

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, rel:torch.Tensor ):

#         # return nn.functional.linear(input, self.weight.clamp(min=0.))

#         # Adapted from https://github.com/NVlabs/E3SM/blob/main/components/eam/src/physics/rrtmgp/slingo.F90#L32
#         # coeffs1 = torch.tensor([2.817e-02, 2.682e-02, 2.264e-02, 1.281e-02], dtype=rel.dtype, device=rel.device)  # A: extinction OD
#         # coeffs2 = torch.tensor([1.305,     1.346,     1.454,     1.641    ], dtype=rel.dtype, device=rel.device)  # B: extinction OD
#         # coeffs3 = torch.tensor([-5.62e-08, -6.94e-06, 4.64e-04,  0.201    ], dtype=rel.dtype, device=rel.device)  # C: single scat albedo
#         # coeffs4 = torch.tensor([1.63e-07,  2.35e-05,  1.24e-03,  7.56e-03 ], dtype=rel.dtype, device=rel.device)  # D: single scat albedo
#         # coeffs5 = torch.tensor([0.829,     0.794,     0.754,     0.826    ], dtype=rel.dtype, device=rel.device)  # E: asymmetry parameter
#         # coeffs6 = torch.tensor([2.482e-03, 4.226e-03, 6.560e-03, 4.353e-03], dtype=rel.dtype, device=rel.device)  # F: asymmetry parameter

#         re_um   = rel.clamp(4.2, 16.0)

#         k       = (self.coeffs1.clamp(min=0.) + self.coeffs2.clamp(min=0.) / re_um)
#         ssa     = (1.0 - self.coeffs3 - re_um * self.coeffs4).clamp(min=0.00001,max=0.999999)
#         g       = (self.coeffs5.clamp(min=0.) + re_um * self.coeffs6.clamp(min=0.)).clamp(min=0.000001,max=0.999999)
#         return k, ssa, g 

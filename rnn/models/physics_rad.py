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
from typing import List, Tuple, Final, Optional

# from settings import disable_compile

# -------------------------------------------- LONGWAVE FUNCTIONS --------------------------------------------

def interpolate_tlev_batchfirst(tlay, play, plev):
    ncol, nlay = tlay.shape
    device = tlay.device
    dtype = tlay.dtype
    # Initialize output arrays
    tlev = torch.zeros(ncol, nlay + 1, dtype=dtype, device=device)
    
    tlev[:,0] = tlay[:,0] + (plev[:,0]-play[:,0])*(tlay[:,1]-tlay[:,0]) / (play[:,1]-play[:,0])
    for ilay in range(1, nlay):
      tlev[:,ilay] = (play[:,ilay-1]*tlay[:,ilay-1]*(plev[:,ilay]-play[:,ilay]) \
            + play[:,ilay]*tlay[:,ilay]*(play[:,ilay-1]-plev[:,ilay])) /  (plev[:,ilay]*(play[:,ilay-1] - play[:,ilay]))
                              
    tlev[:,nlay] = tlay[:,nlay-1] + (plev[:,nlay]-play[:,nlay-1])*(tlay[:,nlay-1]-tlay[:,nlay-2])  \
            / (play[:,nlay-1]-play[:,nlay-2])
                              
    return tlev

def interpolate_tlev_batchlast(tlay, play, plev):
    nlay, ncol = tlay.shape
    device = tlay.device
    dtype = tlay.dtype
    # Initialize output arrays
    tlev = torch.zeros(nlay + 1, ncol, dtype=dtype, device=device)
    
    tlev[0] = tlay[0] + (plev[0]-play[0])*(tlay[1]-tlay[0]) / (play[1]-play[0])
    for ilay in range(1, nlay):
      tlev[ilay] = (play[ilay-1]*tlay[ilay-1]*(plev[ilay]-play[ilay]) \
            + play[ilay]*tlay[ilay]*(play[ilay-1]-plev[ilay])) /  (plev[ilay]*(play[ilay-1] - play[ilay]))
                              
    tlev[nlay] = tlay[nlay-1] + (plev[nlay]-play[nlay-1])*(tlay[nlay-1]-tlay[nlay-2])  \
            / (play[nlay-1]-play[nlay-2])
                              
    return tlev

@torch.compile(dynamic=False)
def outgoing_lw(temp):
    # Stefan-Boltzmann constant (W/m²/K⁴)
    # sigma = 5.670374419e-8
    
    # Assuming emissivity = 1 (blackbody approximation)
    olr_exact = 5.670374419e-8 * torch.pow(temp,4)
    return olr_exact

@torch.compile(dynamic=False)
def reftrans_lw(planck_top, planck_bot, od):
    """
    Calculate longwave transmittance and source terms using Padé approximant method.
    
    This function implements the alternative source computation using a Padé approximant
    for the linear-in-tau solution, following Clough et al. (1992), doi:10.1029/92JD01419, Eq 15.
    This method requires no conditional statements but introduces some approximation error.
    
    Args:
        planck_top (torch.Tensor): Planck function at layer top
        planck_bot (torch.Tensor): Planck function at layer bottom
        od (torch.Tensor): Optical depth
        LwDiffusivity (float): Longwave diffusivity factor (default 1.66)
    
    Returns:
        tuple: (transmittance, source_up, source_dn)
            - source_up (torch.Tensor): Upward emission at layer top 
            - source_dn (torch.Tensor): Downward emission at layer bottom 
            - transmittance (torch.Tensor): Diffuse transmittance
    """
    LwDiffusivity=1.66
    od = LwDiffusivity * od
    trans_lw = torch.exp(-od)
    # Calculate coefficient for Padé approximant (vectorized)
    coeff = 0.2 * od
    # Calculate mean Planck function (vectorized)
    planck_fl = 0.5 * (planck_top + planck_bot)
    # Calculate source terms using Padé approximant (vectorized)
    # one_minus_trans = 1.0 - trans_lw
    # one_plus_coeff = 1.0 + coeff
    source_dn = (1.0 - trans_lw) * (planck_fl + coeff * planck_bot) / (1.0 + coeff)
    source_up = (1.0 - trans_lw) * (planck_fl + coeff * planck_top) / (1.0 + coeff)
    return source_up, source_dn, trans_lw


@torch.compile(dynamic=False)
def lw_solver_noscat_batchlast(trans_lw, source_dn, source_up, source_sfc, emissivity_surf):
    
    nlev = trans_lw.shape[0]
    
    # At top-of-atmosphere there is no diffuse downwelling radiation
    flux_lw_dn0 = torch.zeros_like(emissivity_surf)
    flux_lw_dn = torch.jit.annotate(List[Tensor], [])
    flux_lw_dn += [flux_lw_dn0]

    # Work down through the atmosphere computing the downward fluxes
    # at each half-level (vectorized over columns)
    for jlev in range(nlev):
        # flux_lw_dn[jlev + 1] = (trans_lw[jlev] * flux_lw_dn[jlev].clone()  + 
        #                        source_dn[jlev])
        flux_lw_dn0 = (trans_lw[jlev] * flux_lw_dn0 + source_dn[jlev])
        flux_lw_dn += [flux_lw_dn0]

    # flux_lw_up[nlev] = source_sfc + albedo_surf * flux_lw_dn[nlev]
    #                                              albedo
    flux_lw_up0   = emissivity_surf*source_sfc +  (1-emissivity_surf) * flux_lw_dn[nlev]
    flux_lw_up    = torch.jit.annotate(List[Tensor], [])
    flux_lw_up    += [flux_lw_up0]

    flux_lw_dn = torch.stack(flux_lw_dn)

    # Work back up through the atmosphere computing the upward fluxes
    # at each half-level (vectorized over columns)
    for jlev in range(nlev - 1, -1, -1):
        # flux_lw_up[jlev] = (trans_lw[jlev] * flux_lw_up[jlev + 1].clone()  + 
        #                    source_up[jlev])
        flux_lw_up0 = (trans_lw[jlev] * flux_lw_up0  + source_up[jlev])    
        flux_lw_up += [flux_lw_up0]

    flux_lw_up.reverse()
    flux_lw_up  = torch.stack(flux_lw_up)
    return flux_lw_dn, flux_lw_up

# -------------------------------------------- LONGWAVE FUNCTIONS --------------------------------------------


# -------------------------------------------- SHORTWAVE FUNCTIONS --------------------------------------------

@torch.compile(dynamic=False)
def calc_ref_trans_sw(mu0, od, ssa, asymmetry):
    """
    Two-stream shortwave reflectance and transmittance calculation.
    Implements Meador & Weaver (1980) equations.

    Args:
        mu0:       Cosine of solar zenith angle (ncol*ng) (expanded in outer code)
        od:        Optical depth, shape (ncol*ng)
        ssa:       Single scattering albedo, shape (ncol*ng)
        asymmetry: Asymmetry factor, shape (ncol*ng)

    Returns:
        ref_diff:       Diffuse reflectance.
        trans_diff:     Diffuse transmittance.
        ref_dir:        Direct reflectance.
        trans_dir_diff: Direct-to-diffuse transmittance.
        trans_dir_dir:  Direct unscattered transmittance.
    """
    # eps = torch.finfo(od.dtype).eps
    eps = 1.0e-7

    # ------------------------------------------------------------------ #
    # Unscattered direct transmittance
    # ------------------------------------------------------------------ #
    trans_dir_dir = torch.exp(-od / mu0)

    # ------------------------------------------------------------------ #
    # Two-stream gamma coefficients
    # ------------------------------------------------------------------ #
    factor  = 0.75 * asymmetry
    gamma1  = 2.0  - ssa * (1.25 + factor)
    gamma2  = ssa  * (0.75 - factor)
    gamma3  = 0.5  - mu0 * factor
    gamma4  = 1.0  - gamma3

    # alpha1 / alpha2  (Eqs. 16-17)
    alpha1 = gamma1 * gamma4 + gamma2 * gamma3
    alpha2 = gamma1 * gamma3 + gamma2 * gamma4

    # ------------------------------------------------------------------ #
    # Diffuse reflectance / transmittance  (Eqs. 25-26)
    # ------------------------------------------------------------------ #
    # k_exponent  (Eq. 18) — clamped for numerical safety
    k = torch.sqrt(torch.clamp((gamma1 - gamma2) * (gamma1 + gamma2), min=1.0e-4))

    exponential   = torch.exp(-k * od)
    exponential2  = exponential ** 2
    k_2_exp       = 2.0 * k * exponential

    reftrans_factor = 1.0 / (k + gamma1 + (k - gamma1) * exponential2)

    ref_diff   = gamma2 * (1.0 - exponential2) * reftrans_factor

    zeros=torch.zeros_like(ref_diff)
    trans_diff = torch.clamp(
        k_2_exp * reftrans_factor,
        min=zeros,
        max=1.0 - ref_diff,          # never exceeds 1 − ref_diff
    )
    trans_diff = torch.clamp(trans_diff, min=0.0)

    # ------------------------------------------------------------------ #
    # Direct reflectance / transmittance  (Eqs. 14-15)
    # ------------------------------------------------------------------ #
    k_mu0              = k * mu0
    one_minus_kmu0_sqr = 1.0 - k_mu0 ** 2
    k_gamma3           = k * gamma3
    k_gamma4           = k * gamma4

    # Guard against one_minus_kmu0_sqr ≈ 0 (mirrors Fortran's merge/epsilon)
    safe_denom = torch.where(
        one_minus_kmu0_sqr.abs() > eps,
        one_minus_kmu0_sqr,
        torch.full_like(one_minus_kmu0_sqr, eps),
    )
    # safe_denom = one_minus_kmu0_sqr.abs().clamp(min=eps) * one_minus_kmu0_sqr.sign()

    # reftrans_factor = mu0 * ssa * reftrans_factor / safe_denom
    reftrans_factor = ssa * reftrans_factor / safe_denom

    # Eq. 14
    ref_dir = reftrans_factor * (
            (1.0 - k_mu0) * (alpha2 + k_gamma3)
        - (1.0 + k_mu0) * (alpha2 - k_gamma3) * exponential2
        - k_2_exp * (gamma3 - alpha2 * mu0) * trans_dir_dir
    )

    # Eq. 15 (minus the direct unscattered term)
    trans_dir_diff = reftrans_factor * (
            k_2_exp * (gamma4 + alpha1 * mu0)
        - trans_dir_dir * (
                (1.0 + k_mu0) * (alpha1 + k_gamma4)
            - (1.0 - k_mu0) * (alpha1 - k_gamma4) * exponential2
            )
    )

    # ------------------------------------------------------------------ #
    # Final clipping so that ref_dir + trans_dir_diff ≤ mu0*(1−T_dir_dir)
    # ------------------------------------------------------------------ #
    # max_direct = mu0 * (1.0 - trans_dir_dir)
    max_direct = (1.0 - trans_dir_dir)

    ref_dir        = torch.clamp(ref_dir,        min=zeros, max=max_direct)
    trans_dir_diff = torch.clamp(trans_dir_diff, min=zeros, max=max_direct - ref_dir)

    return ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir


@torch.compile(dynamic=False)
def adding_ica_sw_batchlast_opt(incoming_toa, emissivity_surf_diffuse, emissivity_surf_direct,
                reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir):
        """
        Adding method for shortwave radiation
        Args are torch.Tensors:
          incoming_toa[ncol], emissivity_surf_diffuse[ncol], emissivity_surf_diffuse[ncol],
          reflectance[nlev,ncol], transmittance[nlev,ncol], ref_dir[nlev,ncol], trans_dir_diff[nlev,ncol],
          trans_dir_dir[nlev,ncol]
        Here ncol is a batch dimension without loop dependencies (actually columns x spectral intervals),
        nlev is the number of vertical levels (has loop dependencies)
        Returns:
            tuple: (flux_up, flux_dn_diffuse, flux_dn_direct) each of shape [ncol, nlev+1]
        """
        
        nlev, ncol = reflectance.shape
        device = reflectance.device
        
        # Set surface albedo
        albedo = torch.jit.annotate(List[Tensor], [])
        albedo0 = emissivity_surf_diffuse
        albedo += [albedo0]

        albedodir = torch.jit.annotate(List[Tensor], [])
        albedodir0 = emissivity_surf_direct
        albedodir += [albedodir0]

        # Work up through the atmosphere and compute the albedo of the entire earth/atmosphere system below that half-level
        for jlev in range(nlev-1, -1, -1):  # nlev down to 1 in Fortran indexing

            # comparing ecRad Tripleclouds code to the McICA code, "source" variable  is like fluxdndir*albedodir
            # If we use albedodir instead (like in TripleClouds), we dont need to precompute fluxdndir in a separate loop, so just two vertical loops
            # Adapted from https://github.com/ecmwf-ifs/ecrad/blob/master/radiation/radiation_tripleclouds_sw.F90
            albedodir0 = (ref_dir[jlev] +
                                    (trans_dir_dir[jlev]*albedodir0 + trans_dir_diff[jlev]*albedo0) *
                                    transmittance[jlev] / (1.0 - albedo0 * reflectance[jlev])) #* inv_denom)  
            albedodir  += [albedodir0]  
            
            albedo0 = (reflectance[jlev] + torch.square(transmittance[jlev]) * albedo0  / (1.0 - albedo0 * reflectance[jlev])) 
            albedo  += [albedo0]

        albedo.reverse(); albedodir.reverse()
        
        # At top-of-atmosphere, all upwelling radiation is due to scattering by the direct beam below that level
        fluxup = incoming_toa*albedodir[0]
        flux_up = torch.jit.annotate(List[Tensor], [])
        flux_up += [fluxup]

        fluxdndir = incoming_toa
        flux_dn_direct = torch.jit.annotate(List[Tensor], [])
        flux_dn_direct += [fluxdndir]

        # At top-of-atmosphere there is no diffuse downwelling radiation
        fluxdndiff = torch.zeros_like(incoming_toa)
        flux_dn_diffuse = torch.jit.annotate(List[Tensor], [])
        flux_dn_diffuse += [fluxdndiff]

        # Work back down through the atmosphere computing the fluxes at each half-level
        for jlev in range(nlev):  # 1 to nlev in Fortran indexing

            fluxdndiff = (transmittance[jlev]*fluxdndiff + fluxdndir 
                    * (transmittance[jlev]*albedodir[jlev + 1]*reflectance[jlev]  + trans_dir_diff[jlev] ) 
                    / (1.0-  reflectance[jlev]*albedo[jlev + 1])) 

            # flux_dn_direct[jlev + 1] = fluxdndir * trans_dir_dir[jlev,:]
            fluxdndir =  fluxdndir * trans_dir_dir[jlev,:]
            flux_dn_direct  += [fluxdndir]
            flux_dn_diffuse += [fluxdndiff]

            # flux_up[jlev+1] =  fluxdndir*albedodir[jlev+1] + fluxdndiff* albedo[jlev + 1]
            fluxup = fluxdndir*albedodir[jlev+1] + fluxdndiff* albedo[jlev + 1]
            flux_up += [fluxup]       
            # Apply cosine correction to direct flux
            # flux_dn_direct[:, jlev] = fluxdndir * cos_sza
        
        flux_dn_direct  = torch.stack(flux_dn_direct)
        flux_dn_diffuse = torch.stack(flux_dn_diffuse)
        flux_up = torch.stack(flux_up)

        # Final cosine correction for surface direct flux
        # flux_dn_direct[:, nlev] = flux_dn_direct[:, nlev] * cos_sza
        
        return flux_up, flux_dn_diffuse, flux_dn_direct

# Claude's failed attempt at further optimizing the kernel - actually slower
@torch.compile(dynamic=False, fullgraph=True)
def adding_ica_sw_batchlast_opt_v2(
    incoming_toa,
    emissivity_surf_diffuse,
    emissivity_surf_direct,
    reflectance,
    transmittance,
    ref_dir,
    trans_dir_diff,
    trans_dir_dir,
):
    nlev, ncol = reflectance.shape

    # --- Upward sweep ---
    albedo    = [emissivity_surf_diffuse]
    albedodir = [emissivity_surf_direct]

    albedo0    = emissivity_surf_diffuse
    albedodir0 = emissivity_surf_direct

    for jlev in range(nlev - 1, -1, -1):
        R  = reflectance[jlev]
        T  = transmittance[jlev]

        inv_denom  = 1.0 / (1.0 - albedo0 * R)   # computed ONCE, shared below

        albedodir0 = (ref_dir[jlev] + (trans_dir_dir[jlev] * albedodir0
                      + trans_dir_diff[jlev] * albedo0) * T * inv_denom)
        albedodir += [albedodir0]

        albedo0    = (R + T * T * albedo0 * inv_denom)
        albedo    += [albedo0]


    albedo.reverse()
    albedodir.reverse()

    # --- Downward sweep ---
    fluxdndir  = incoming_toa
    fluxdndiff = torch.zeros(ncol, dtype=reflectance.dtype, device=reflectance.device)
    fluxup     = incoming_toa * albedodir[0]

    flux_up         = [fluxup]
    flux_dn_direct  = [fluxdndir]
    flux_dn_diffuse = [fluxdndiff]

    for jlev in range(nlev):
        R     = reflectance[jlev]
        T     = transmittance[jlev]
        alb1  = albedo[jlev + 1]
        adir1 = albedodir[jlev + 1]

        inv_denom  = 1.0 / (1.0 - R * alb1)      # computed ONCE, shared below

        fluxdndiff = (T * fluxdndiff + fluxdndir
                      * (T * adir1 * R + trans_dir_diff[jlev]) * inv_denom)
        fluxdndir  = fluxdndir * trans_dir_dir[jlev]

        flux_dn_direct  += [fluxdndir]
        flux_dn_diffuse += [fluxdndiff]
        flux_up         += [fluxdndir * adir1 + fluxdndiff * alb1]

    return (
        torch.stack(flux_up),
        torch.stack(flux_dn_diffuse),
        torch.stack(flux_dn_direct),
    )
    
@torch.jit.script
def adding_ica_sw_inference(
    incoming_toa: Tensor,
    emissivity_surf_diffuse: Tensor,
    emissivity_surf_direct: Tensor,
    reflectance: Tensor,
    transmittance: Tensor,
    ref_dir: Tensor,
    trans_dir_diff: Tensor,
    trans_dir_dir: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:

    nlev, ncol = reflectance.shape

    # --- Upward sweep ---
    # Only keep current-level scalars; no list accumulation needed.
    # BUT: the downward sweep still needs albedo[jlev+1] at every level,
    # so we cannot avoid storing the full arrays.
    # We CAN avoid the Python list + reverse + stack overhead by writing
    # directly into pre-allocated tensors — safe here because autograd
    # is not running, so no version counter issues.
    albedo    = torch.empty(nlev + 1, ncol, dtype=reflectance.dtype,
                             device=reflectance.device)
    albedodir = torch.empty(nlev + 1, ncol, dtype=reflectance.dtype,
                             device=reflectance.device)

    albedo[0]    = emissivity_surf_diffuse
    albedodir[0] = emissivity_surf_direct

    alb0  = emissivity_surf_diffuse
    adir0 = emissivity_surf_direct

    for k in range(nlev):
        jlev = nlev - 1 - k
        R  = reflectance[jlev]
        T  = transmittance[jlev]
        inv_denom = 1.0 / (1.0 - alb0 * R)
        adir0 = ref_dir[jlev] + (trans_dir_dir[jlev] * adir0
                + trans_dir_diff[jlev] * alb0) * T * inv_denom
        alb0  = R + T * T * alb0 * inv_denom
        albedo   [k + 1] = alb0
        albedodir[k + 1] = adir0

    # albedo   [0]    = surface,  albedo   [nlev] = TOA (bottom-up order)
    # For downward sweep: albedo below level jlev (0=TOA) = albedo[nlev - jlev]
    # (same indexing as the fixed Triton kernel)

    # --- Downward sweep ---
    flux_up         = torch.empty(nlev + 1, ncol, dtype=reflectance.dtype,
                                   device=reflectance.device)
    flux_dn_diffuse = torch.empty(nlev + 1, ncol, dtype=reflectance.dtype,
                                   device=reflectance.device)
    flux_dn_direct  = torch.empty(nlev + 1, ncol, dtype=reflectance.dtype,
                                   device=reflectance.device)

    fluxdndir  = incoming_toa
    fluxdndiff = torch.zeros(ncol, dtype=reflectance.dtype,
                              device=reflectance.device)

    flux_up        [0] = incoming_toa * albedodir[nlev]   # TOA = slot nlev
    flux_dn_direct [0] = fluxdndir
    flux_dn_diffuse[0] = fluxdndiff

    for jlev in range(nlev):
        R     = reflectance[jlev]
        T     = transmittance[jlev]
        # albedo below this level = slot (nlev - jlev)
        below = nlev - jlev
        alb1  = albedo   [below]
        adir1 = albedodir[below]
        inv_denom  = 1.0 / (1.0 - R * alb1)
        fluxdndiff = (T * fluxdndiff
                      + fluxdndir * (T * adir1 * R + trans_dir_diff[jlev])
                      * inv_denom)
        fluxdndir  = fluxdndir * trans_dir_dir[jlev]
        flux_dn_direct [jlev + 1] = fluxdndir
        flux_dn_diffuse[jlev + 1] = fluxdndiff
        flux_up        [jlev + 1] = fluxdndir * adir1 + fluxdndiff * alb1

    return flux_up, flux_dn_diffuse, flux_dn_direct

def adding_ica_sw(
    incoming_toa, emissivity_surf_diffuse, emissivity_surf_direct,
    reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir
):
    if torch.is_grad_enabled():
        # print("calling normal adding!")
        return adding_ica_sw_batchlast_opt(
            incoming_toa, emissivity_surf_diffuse, emissivity_surf_direct,
            reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir
        )
    else:
        # print("calling inference adding!")
        return adding_ica_sw_inference(
            incoming_toa, emissivity_surf_diffuse, emissivity_surf_direct,
            reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir
        )

@torch.compile(dynamic=False)
def adding_tc_sw_batchlast_opt(incoming_toa, emissivity_surf_diffuse, emissivity_surf_direct,
                reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir, V, ncol_subgrid):
        """
        Adding method for shortwave radiation, experimental TripleClouds version
        Requires overlap matrix V

        Args are torch.Tensors:
          incoming_toa[ncol], emissivity_surf_diffuse[ncol], emissivity_surf_diffuse[ncol],
          reflectance[nlev,ncol], transmittance[nlev,ncol], ref_dir[nlev,ncol], trans_dir_diff[nlev,ncol],
          trans_dir_dir[nlev,ncol], V[nlev,ncol]

        Returns:
            tuple: (flux_up, flux_dn_diffuse, flux_dn_direct) each of shape [ncol, nlev+1]
        """
        
        nlev, ncol = reflectance.shape
        device = reflectance.device
        
        # Set surface albedo
        albedo = torch.jit.annotate(List[Tensor], [])
        albedo0 = emissivity_surf_diffuse
        albedo += [albedo0]

        albedodir = torch.jit.annotate(List[Tensor], [])
        albedodir0 = emissivity_surf_direct
        albedodir += [albedodir0]

        # Work up through the atmosphere and compute the albedo of the entire earth/atmosphere system below that half-level
        for jlev in range(nlev-1, -1, -1):  # nlev down to 1 in Fortran indexing

            # comparing ecRad Tripleclouds code to the McICA code, "source" variable  is like fluxdndir*albedodir
            # If we use albedodir instead (like in TripleClouds), we dont need to precompute fluxdndir in a separate loop, so just two vertical loops
            # Adapted from https://github.com/ecmwf-ifs/ecrad/blob/master/radiation/radiation_tripleclouds_sw.F90
            albedodir0 = (ref_dir[jlev] +
                                    (trans_dir_dir[jlev]*albedodir0 + trans_dir_diff[jlev]*albedo0) *
                                    transmittance[jlev] / (1.0 - albedo0 * reflectance[jlev])) #*V[jlev] #* inv_denom)  
            
            albedodir0 = albedodir0.view(-1,ncol_subgrid)
            Vmat = V[jlev].view(-1,ncol_subgrid,ncol_subgrid)
            # for nreg=3:
            # A[:,1] =  A[:,1]*V[:,1,1] + A[:,2]*V[:,2,1] + A[:,3]*V[:,3,1]
            # A[:,2] =  A[:,1]*V[:,1,2] + A[:,2]*V[:,2,2] + A[:,3]*V[:,3,2]
            # A[:,3] =  A[:,1]*V[:,1,3] + A[:,2]*V[:,2,3] + A[:,3]*V[:,3,3]
            albedodir0 = torch.bmm(albedodir0.unsqueeze(1), Vmat).squeeze(1)
            albedodir0 = albedodir0.view(-1)
            albedodir  += [albedodir0]  
            
            albedo0 = (reflectance[jlev] + torch.square(transmittance[jlev]) * albedo0  / (1.0 - albedo0 * reflectance[jlev])) #*V[jlev]
            albedo0 = albedo0.view(-1,ncol_subgrid)
            albedo0 = torch.bmm(albedo0.unsqueeze(1), Vmat).squeeze(1)
            albedo0 = albedo0.view(-1)


            albedo  += [albedo0]

            # TripleClouds :
            # ng_nreg = self.nreg//3 
            # A[:,1,lev]            = A[:,1,jlev]*V[1,1] + A[:,2,jlev]*V[2,1] + A[:,3,jlev]*V[3,1]
            # A[:,2,jlev]           = A[:,1,jlev]*V[1,2] + A[:,2,jlev]*V[2,2] + A[:,3,jlev]*V[3,2]
            # A[:,3,jlev]           = A[:,1,jlev]*V[1,3] + A[:,2,jlev]*V[2,3] + A[:,3,jlev]*V[3,3]         
            # A[jlev]           = A[jlev]*V[jlev]

        albedo.reverse(); albedodir.reverse()
        
        # At top-of-atmosphere, all upwelling radiation is due to scattering by the direct beam below that level
        fluxup = incoming_toa*albedodir[0]
        flux_up = torch.jit.annotate(List[Tensor], [])
        flux_up += [fluxup]

        fluxdndir = incoming_toa
        flux_dn_direct = torch.jit.annotate(List[Tensor], [])
        flux_dn_direct += [fluxdndir]

        # At top-of-atmosphere there is no diffuse downwelling radiation
        fluxdndiff = torch.zeros_like(incoming_toa)
        flux_dn_diffuse = torch.jit.annotate(List[Tensor], [])
        flux_dn_diffuse += [fluxdndiff]

        # Work back down through the atmosphere computing the fluxes at each half-level
        for jlev in range(nlev):  # 1 to nlev in Fortran indexing

            fluxdndiff = (transmittance[jlev]*fluxdndiff + fluxdndir 
                    * (transmittance[jlev]*albedodir[jlev + 1]*reflectance[jlev]  + trans_dir_diff[jlev] ) 
                    / (1.0-  reflectance[jlev]*albedo[jlev + 1])) 

            # flux_dn_direct[jlev + 1] = fluxdndir * trans_dir_dir[jlev,:]
            fluxdndir =  fluxdndir * trans_dir_dir[jlev,:]
            flux_dn_direct  += [fluxdndir]
            flux_dn_diffuse += [fluxdndiff]

            # flux_up[jlev+1] =  fluxdndir*albedodir[jlev+1] + fluxdndiff* albedo[jlev + 1]
            fluxup = fluxdndir*albedodir[jlev+1] + fluxdndiff* albedo[jlev + 1]
            flux_up += [fluxup]       
            # Apply cosine correction to direct flux
            # flux_dn_direct[:, jlev] = fluxdndir * cos_sza
        
        flux_dn_direct  = torch.stack(flux_dn_direct)
        flux_dn_diffuse = torch.stack(flux_dn_diffuse)
        flux_up = torch.stack(flux_up)

        # Final cosine correction for surface direct flux
        # flux_dn_direct[:, nlev] = flux_dn_direct[:, nlev] * cos_sza
        
        return flux_up, flux_dn_diffuse, flux_dn_direct
    


# -------------------------------------------- SHORTWAVE FUNCTIONS --------------------------------------------



def stratified_sample(p, G, shuffle=True):
    """
    Stratified sampling: assign G spectral points among N sub-grid states proportional to p

    Args:
        p:       (B, N) - area fractions, summing to 1 along last dim
        G:       int          - number of spectral points
        shuffle: bool         - shuffle assignment within batch elements

    Returns:
        state_indices: (B, G) - (shuffled) index array which can be used to gather from x(B,N) into x(B,G) 
    """
    B, N = p.shape
    device = p.device

    # --- Step 1: Compute counts via largest remainder method ---
    # This ensures counts sum exactly to G
    exact = p * G                                        # (B, N)
    floors = exact.floor().long()                        # (B, N)
    remainders = exact - floors.float()                  # (B, N)
    deficit = G - floors.sum(dim=-1, keepdim=True)       # (B, 1) — how many left to assign

    # Assign remaining counts to states with largest remainders
    _, remainder_ranks = remainders.sort(dim=-1, descending=True)  # (B, N)
    rank_threshold = remainder_ranks.argsort(dim=-1)               # (B, N)
    bonus = (rank_threshold < deficit).long()                      # (B, N)
    counts = floors + bonus                                        # (B, N), sums to G

    # --- Step 2: Build index array for each batch element ---
    # Vectorised index construction (no Python loop)
    # Place a 1 at the start of each state's block, then cumsum to fill
    counts_clamped = counts.clamp(min=0)
    starts = torch.zeros(B, G + 1, dtype=torch.long, device=device)
    offsets = counts.cumsum(dim=-1)                 # (B, N) — end positions

    # Scatter state transitions into (B, G+1), then cumsum gives state index
    src = torch.ones(B, N, dtype=torch.long, device=device)
    starts.scatter_add_(dim=-1, index=offsets, src=src)
    state_indices = starts[:, :G].cumsum(dim=-1)  # (B, G)  
    # print("p 100, 40 inds", state_indices[100*40,:], "p", p[100*40,:])

    # --- Step 3: Shuffle spectral dimension ---
    if shuffle:
        idx = torch.randperm(G)
        state_indices = state_indices[:, idx]
    # --- Step 4: Gather x values ---
    # x_sampled = torch.gather(x_flat, dim=-1, state_indices)             # (B, G)

    return state_indices

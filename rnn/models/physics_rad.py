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

    Args:   (variable dimensions don't matter because all operations are element wise)
        mu0:       Cosine of solar zenith angle (nlev,ncol,ng) (expanded view)
        od:        Optical depth, shape (nlev,ncol,ng) 
        ssa:       Single scattering albedo, shape (nlev,ncol,ng) 
        asymmetry: Asymmetry factor, shape (nlev,ncol,ng) 

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
    gamma1 = (8 - ssa*(5 + 3*asymmetry)) * 0.25
    gamma2 = 3*(ssa*(1 - asymmetry)) * 0.25
    gamma3 = (2 - 3*mu0*asymmetry) * 0.25
    gamma4  = 1.0  - gamma3

    # alpha1 / alpha2  (Eqs. 16-17)
    alpha1 = gamma1 * gamma4 + gamma2 * gamma3
    alpha2 = gamma1 * gamma3 + gamma2 * gamma4

    # ------------------------------------------------------------------ #
    # Diffuse reflectance / transmittance  (Eqs. 25-26)
    # ------------------------------------------------------------------ #
    # k_exponent  (Eq. 18) — clamped for numerical safety
    k = torch.sqrt(torch.clamp((gamma1 - gamma2) * (gamma1 + gamma2), min=1.0e-4)) # 1e-4 TUNED FOR SINGLE PRECISION!

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


# @torch.jit.script
# @torch.compile(dynamic=False, backend="inductor")
# @torch.compile(dynamic=False,mode="max-autotune-no-cudagraphs")
@torch.compile(dynamic=False)
def adding_ica_sw_batchlast_opt(incoming_toa, albedo_surf_diffuse, albedo_surf_direct,
                R, T, ref_dir, T_dir_diff, T_dir_dir):
        """
        Adding method for shortwave radiation. 
        Adapted from ecRad-TripleClouds to use just two vertical loops instead of three as in RTE. 
        Args are torch.Tensors:
          incoming_toa[nbatch], albedo_surf_diffuse[nbatch], albedo_surf_direct[nbatch],
          reflectance[nlev,nbatch], transmittance[nlev,nbatch], ref_dir[nlev,nbatch], trans_dir_diff[nlev,nbatch],
          trans_dir_dir[nlev,nbatch]
        Here nlev is the contiguous dimension in memory and nbatch (=ng*ncol) is a batch dimension without loop dependencies, 
            where spectral (ng) and column (ncol) dimensions should have been collapsed before calling this function. 
        This should be optimal for GPU; for CPU it *may* be better to have (ng,nlev,ncol) (in Python row major notation)
            where SIMD vectorization is used for the innermost ng and multithreading is used for outermost ncol. 
        Returns:
            tuple: (flux_up, flux_dn_diffuse, flux_dn_direct) each of shape [nbatch, nlev+1]
        """
        
        nlev, nbatch = R.shape
        device = R.device
        
        # Set surface albedo
        albedo = torch.jit.annotate(List[Tensor], [])
        albedo0 = albedo_surf_diffuse
        albedo += [albedo0]

        albedodir = torch.jit.annotate(List[Tensor], [])
        albedodir0 = albedo_surf_direct
        albedodir += [albedodir0]

        # Work up through the atmosphere and compute the albedo of the entire earth/atmosphere system below that half-level
        for jlev in range(nlev-1, -1, -1):  # nlev down to 1 in Fortran indexing

            # comparing ecRad Tripleclouds code to the McICA code, "source" variable  is like fluxdndir*albedodir
            # If we use albedodir instead (like in TripleClouds), we dont need to precompute fluxdndir in a separate loop, so just two vertical loops
            # Adapted from https://github.com/ecmwf-ifs/ecrad/blob/master/radiation/radiation_tripleclouds_sw.F90
            inv_denom = 1.0/(1.0 - albedo0 * R[jlev])
            albedodir0 = ref_dir[jlev] + (T_dir_dir[jlev]*albedodir0 + T_dir_diff[jlev]*albedo0) *T[jlev]* inv_denom
            albedodir  += [albedodir0]  
            
            albedo0 = R[jlev] + torch.square(T[jlev]) * albedo0  * inv_denom #/ (1.0 - albedo0 * R[jlev])
            albedo  += [albedo0]

        # Reverse arrays because next loop will go from top-of-atmosphere to surface
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

            fluxdndiff = (T[jlev]*fluxdndiff 
                + fluxdndir * (T[jlev]*albedodir[jlev+1]*R[jlev] + T_dir_diff[jlev])) / (1.0 - R[jlev]*albedo[jlev+1])

            fluxdndir =  fluxdndir * T_dir_dir[jlev,:]
            # Apply cosine correction to direct flux..NOT HERE, already done to TOA incoming flux
            # flux_dn_direct = fluxdndir * cos_sza
            flux_dn_direct  += [fluxdndir]
            flux_dn_diffuse += [fluxdndiff]

            fluxup = fluxdndir*albedodir[jlev+1] + fluxdndiff* albedo[jlev + 1]
            flux_up += [fluxup]       
        
        flux_dn_direct  = torch.stack(flux_dn_direct)
        flux_dn_diffuse = torch.stack(flux_dn_diffuse)
        flux_up = torch.stack(flux_up)

        return flux_up, flux_dn_diffuse, flux_dn_direct

# @torch.jit.script
@torch.compile(dynamic=False)
def adding_ica_sw_inference(
    incoming_toa: Tensor,
    albedo_surf_diffuse: Tensor,
    albedo_surf_direct: Tensor,
    reflectance: Tensor,
    transmittance: Tensor,
    ref_dir: Tensor,
    trans_dir_diff: Tensor,
    trans_dir_dir: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:

    nlev, nbatch = reflectance.shape

    # --- Upward sweep ---
    # Only keep current-level scalars; no list accumulation needed.
    # BUT: the downward sweep still needs albedo[jlev+1] at every level,
    # so we cannot avoid storing the full arrays.
    # We CAN avoid the Python list + reverse + stack overhead by writing
    # directly into pre-allocated tensors — safe here because autograd
    # is not running, so no version counter issues.
    albedo    = torch.empty(nlev + 1, nbatch, dtype=reflectance.dtype,
                             device=reflectance.device)
    albedodir = torch.empty(nlev + 1, nbatch, dtype=reflectance.dtype,
                             device=reflectance.device)

    albedo[0]    = albedo_surf_diffuse
    albedodir[0] = albedo_surf_direct

    alb0  = albedo_surf_diffuse
    adir0 = albedo_surf_direct

    for k in range(nlev):
        jlev = nlev - 1 - k
        R  = reflectance[jlev]
        T  = transmittance[jlev]
        inv_denom = 1.0 / (1.0 - alb0 * R)
        adir0 = ref_dir[jlev] + (trans_dir_dir[jlev] * adir0 + trans_dir_diff[jlev] * alb0) * T * inv_denom
        alb0  = R + T * T * alb0 * inv_denom
        albedo   [k + 1] = alb0
        albedodir[k + 1] = adir0

    # albedo   [0]    = surface,  albedo   [nlev] = TOA (bottom-up order)
    # For downward sweep: albedo below level jlev (0=TOA) = albedo[nlev - jlev]
    # (same indexing as the fixed Triton kernel)

    # --- Downward sweep ---
    flux_up         = torch.empty(nlev + 1, nbatch, dtype=reflectance.dtype,
                                   device=reflectance.device)
    flux_dn_diffuse = torch.empty(nlev + 1, nbatch, dtype=reflectance.dtype,
                                   device=reflectance.device)
    flux_dn_direct  = torch.empty(nlev + 1, nbatch, dtype=reflectance.dtype,
                                   device=reflectance.device)

    fluxdndir  = incoming_toa
    fluxdndiff = torch.zeros(nbatch, dtype=reflectance.dtype,
                              device=reflectance.device)

    flux_up        [0] = incoming_toa * albedodir[nlev]   # TOA = slot nlev
    flux_dn_direct [0] = fluxdndir
    flux_dn_diffuse[0] = fluxdndiff

    for jlev in range(nlev):
        R     = reflectance[jlev]
        T     = transmittance[jlev]
        below = nlev - (jlev + 1)
        alb1  = albedo   [below]
        adir1 = albedodir[below]
        inv_denom  = 1.0 / (1.0 - R * alb1)
        fluxdndiff = ((T * fluxdndiff  + fluxdndir * (T * adir1 * R + trans_dir_diff[jlev])) * inv_denom)
        fluxdndir  = fluxdndir * trans_dir_dir[jlev]
        flux_dn_direct [jlev + 1] = fluxdndir
        flux_dn_diffuse[jlev + 1] = fluxdndiff
        flux_up        [jlev + 1] = fluxdndir * adir1 + fluxdndiff * alb1

    return flux_up, flux_dn_diffuse, flux_dn_direct

def adding_ica_sw(
    incoming_toa, albedo_surf_diffuse, albedo_surf_direct,
    reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir
):
    if torch.is_grad_enabled():
        # print("calling normal adding!")
        return adding_ica_sw_batchlast_opt(
            incoming_toa, albedo_surf_diffuse, albedo_surf_direct,
            reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir
        )
    else:
        # print("calling inference adding!")
        return adding_ica_sw_inference(
            incoming_toa, albedo_surf_diffuse, albedo_surf_direct,
            reflectance, transmittance, ref_dir, trans_dir_diff, trans_dir_dir
        )

@torch.compile(dynamic=False)
def adding_tc_sw_batchlast_opt(
        incoming_toa: Tensor,
        albedo_surf_diffuse: Tensor,
        albedo_surf_direct: Tensor,
        R: Tensor,
        T: Tensor,
        ref_dir: Tensor,
        T_dir_diff: Tensor,
        T_dir_dir: Tensor,
        V: Tensor, 
        nreg: int) -> Tuple[Tensor, Tensor, Tensor]:

        """
        Adding method for shortwave radiation, experimental TripleClouds version
        Requires overlap matrix V

        Args are torch.Tensors:
          incoming_toa[nbatch], albedo_surf_diffuse[nbatch], albedo_surf_diffuse[nbatch],
          R[nlev,nbatch], T[nlev,nbatch], ref_dir[nlev,nbatch], T_dir_diff[nlev,nbatch],
          T_dir_dir[nlev,nbatch], V[nlev,nbatch]

        Not done, this bit is missing!:
        https://github.com/peterukk/ecrad-opt/blob/clean_no_opt_testing/radiation/radiation_tripleclouds_sw.F90#L624

        Returns:
            tuple: (flux_up, flux_dn_diffuse, flux_dn_direct) each of shape [nbatch, nlev+1]
        """

        nlev, nbatch = R.shape
        device = R.device
        
        # Set surface albedo
        albedo = torch.jit.annotate(List[Tensor], [])
        albedo0 = albedo_surf_diffuse
        albedo += [albedo0]

        albedodir = torch.jit.annotate(List[Tensor], [])
        albedodir0 = albedo_surf_direct
        albedodir += [albedodir0]

        # Work up through the atmosphere and compute the albedo of the entire earth/atmosphere system below that half-level
        for jlev in range(nlev-1, -1, -1):  # nlev down to 1 in Fortran indexing
            albedodir0 = (ref_dir[jlev] +
                                    (T_dir_dir[jlev]*albedodir0 + T_dir_diff[jlev]*albedo0) *
                                    T[jlev] / (1.0 - albedo0 * R[jlev]))  
            
            albedodir0 = albedodir0.view(-1,1,nreg)
            Vmat = V[jlev].view(-1,nreg,nreg)
            # Account for cloud overlap when computing albedo, using direction overlap matrix V for nreg=3 this is:
            # A[:,1] =  A[:,1]*V[:,1,1] + A[:,2]*V[:,2,1] + A[:,3]*V[:,3,1]
            # A[:,2] =  A[:,1]*V[:,1,2] + A[:,2]*V[:,2,2] + A[:,3]*V[:,3,2]
            # A[:,3] =  A[:,1]*V[:,1,3] + A[:,2]*V[:,2,3] + A[:,3]*V[:,3,3]
            albedodir0 = torch.bmm(albedodir0, Vmat)
            albedodir0 = albedodir0.view(-1)
            albedodir  += [albedodir0]  
            
            albedo0 = R[jlev] + torch.square(T[jlev]) * albedo0  / (1.0 - albedo0 * R[jlev]) 
            albedo0 = albedo0.view(-1,1,nreg)
            albedo0 = torch.bmm(albedo0, Vmat)
            albedo0 = albedo0.view(-1)


            albedo  += [albedo0]


        # Reverse arrays because next loop will go from top-of-atmosphere to surface
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

            fluxdndiff = (T[jlev]*fluxdndiff 
                + fluxdndir * (T[jlev]*albedodir[jlev+1]*R[jlev] + T_dir_diff[jlev])) / (1.0 - R[jlev]*albedo[jlev+1])

            fluxdndir =  fluxdndir * T_dir_dir[jlev,:]
            # Apply cosine correction to direct flux..NOT HERE, already done to TOA incoming flux
            # flux_dn_direct = fluxdndir * cos_sza
            Vmat = V[jlev].view(-1,nreg,nreg)
            fluxdndir   = fluxdndir.view(-1,nreg,1)
            fluxdndiff  = fluxdndiff.view(-1,nreg,1)
            fluxdndir   = torch.bmm(Vmat, fluxdndir)
            fluxdndiff  = torch.bmm(Vmat, fluxdndiff)
            fluxdndir   = fluxdndir.view(-1)
            fluxdndiff  = fluxdndiff.view(-1)

            flux_dn_direct  += [fluxdndir]
            flux_dn_diffuse += [fluxdndiff]

            fluxup = fluxdndir*albedodir[jlev+1] + fluxdndiff* albedo[jlev + 1]
            flux_up += [fluxup]       
        
        flux_dn_direct  = torch.stack(flux_dn_direct)
        flux_dn_diffuse = torch.stack(flux_dn_diffuse)
        flux_up = torch.stack(flux_up)
        
        return flux_up, flux_dn_diffuse, flux_dn_direct
    


# -------------------------------------------- SHORTWAVE FUNCTIONS --------------------------------------------

def stratified_sample(p: torch.Tensor, G: int) -> torch.Tensor:
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
    # if shuffle:
    #     idx = torch.randperm(G)
    #     state_indices = state_indices[:, idx]
    # --- Step 4: Gather x values ---
    # x_sampled = torch.gather(x_flat, dim=-1, state_indices)             # (B, G)

    return state_indices

"""
PyTorch port of calc_beta_overlap_matrix and calc_overlap_matrices_nocol
from Shonk et al. (2010) / Hogan & Illingworth (2000).
 
Conventions
-----------
* nreg = 3 (hard-coded; all region loops are fully unrolled).
* Tensor layout: (nreg, nlev, nbatch)  —  nreg is the innermost logical
  axis but outermost memory axis so that slices like t[0], t[1], t[2]
  are contiguous, enabling fast batched arithmetic on GPU.
* Level index 0 = top-of-atmosphere, index nlev-1 = lowest model level,
  matching the Fortran convention (1-based there, 0-based here).
* The output v_matrix has shape (nreg, nreg, nlev+1, nbatch):
    v_matrix[jlower, jupper, jlev, :] = overlap_matrix[jupper, jlower]
                                         / frac_upper[jupper]
  i.e. the same definition as the Fortran.
"""
 
# ---------------------------------------------------------------------------
# Internal helper: batched beta overlap matrix
# ---------------------------------------------------------------------------
@torch.compile(dynamic=False)
def _calc_beta_overlap_matrix(
    op: Tensor,           # (3, nbatch)
    frac_upper: Tensor,   # (3, nbatch)
    frac_lower: Tensor,   # (3, nbatch)
    frac_threshold: float,
) -> Tensor:
    """
    Compute the (3, 3, nbatch) overlap matrix for one interface.
 
    Implements Eq. (A1)-(A2) of Shonk et al. (2010):
 
        overlap_matrix[i, j] =
            op_x_frac_min[i] * delta(i,j)          (maximum part)
          + factor * (frac_lower[j] - op_x_frac_min[j])
                   * (frac_upper[i] - op_x_frac_min[i])  (random part)
 
    where  op_x_frac_min[i] = op[i] * min(frac_upper[i], frac_lower[i])
    and    factor = 1 / (1 - sum_i op_x_frac_min[i]).
 
    Parameters
    ----------
    op, frac_upper, frac_lower : (3, nbatch) tensors
    frac_threshold : scalar float
 
    Returns
    -------
    overlap_matrix : (3, 3, nbatch)  —  first index = jupper, second = jlower
    """
    # --- op_x_frac_min[i] = op[i] * min(frac_upper[i], frac_lower[i]) ---
    # Unrolled for nreg = 3
    oxm0 = op[0] * torch.minimum(frac_upper[0], frac_lower[0])  # (nbatch,)
    oxm1 = op[1] * torch.minimum(frac_upper[1], frac_lower[1])
    oxm2 = op[2] * torch.minimum(frac_upper[2], frac_lower[2])
 
    # --- denominator = 1 - sum(op_x_frac_min) ---
    denominator = 1.0 - (oxm0 + oxm1 + oxm2)                    # (nbatch,)
 
    # --- random part: factor * outer((frac_upper - oxm), (frac_lower - oxm)) ---
    valid = denominator >= frac_threshold                         # (nbatch,) bool
    factor = torch.where(valid, 1.0 / denominator.clamp(min=frac_threshold),
                         torch.zeros_like(denominator))           # (nbatch,)
 
    # residual fracs for upper and lower (3, nbatch)
    ru0 = frac_upper[0] - oxm0
    ru1 = frac_upper[1] - oxm1
    ru2 = frac_upper[2] - oxm2
 
    rl0 = frac_lower[0] - oxm0
    rl1 = frac_lower[1] - oxm1
    rl2 = frac_lower[2] - oxm2
 
    # Random part (9 elements, fully unrolled):
    # m[jupper, jlower] = factor * ru[jupper] * rl[jlower]
    # but only where denominator >= frac_threshold (factor is already zeroed)
    m00 = factor * ru0 * rl0
    m01 = factor * ru0 * rl1
    m02 = factor * ru0 * rl2
    m10 = factor * ru1 * rl0
    m11 = factor * ru1 * rl1
    m12 = factor * ru1 * rl2
    m20 = factor * ru2 * rl0
    m21 = factor * ru2 * rl1
    m22 = factor * ru2 * rl2
 
    # Add maximum part on diagonal:
    # overlap_matrix[jreg, jreg] += op_x_frac_min[jreg]
    m00 = m00 + oxm0
    m11 = m11 + oxm1
    m22 = m22 + oxm2
 
    # Stack into (3, 3, nbatch): first dim = jupper, second = jlower
    row0 = torch.stack([m00, m01, m02], dim=0)   # (3, nbatch)
    row1 = torch.stack([m10, m11, m12], dim=0)
    row2 = torch.stack([m20, m21, m22], dim=0)
    overlap_matrix = torch.stack([row0, row1, row2], dim=0)      # (3, 3, nbatch)
 
    return overlap_matrix
 
 
# ---------------------------------------------------------------------------
# Public function: batched directional overlap matrices
# ---------------------------------------------------------------------------
@torch.compile(dynamic=False)
def calc_overlap_matrices(
    region_fracs: Tensor,
    overlap_param: Tensor,
    cloud_fraction_threshold: float = 1.0e-20,
) -> Tensor:
    """
    Compute the downward (v) and upward (u) directional overlap matrices
    for all interfaces in a batched column set.
 
    This is a PyTorch port of ``calc_overlap_matrices_nocol`` (nreg=3,
    all region loops unrolled).
 
    Parameters
    ----------
    region_fracs : Tensor, shape (nlev, 3, nbatch)
        Area fraction of each region at each model level.
        Region 0 is clear sky; regions 1 and 2 are cloudy.
        Level 0 = top-of-atmosphere.
 
    overlap_param : Tensor, shape (nlev-1, nbatch)
        beta (Shonk et al. 2010) overlap parameter for each interior interface.
 
    cloud_fraction_threshold : float, optional
        Regions smaller than this fraction are ignored (default 1e-20).
 
    Returns
    -------
    v_matrix : Tensor, shape (3, 3, nlev+1, nbatch)
        Downward directional overlap matrix.
        ``v_matrix[jlower, jupper, jlev, :]`` gives the fraction of
        upwelling flux from region ``jupper`` above interface ``jlev``
        that enters region ``jlower`` below that interface.
 
    Notes
    -----
    * Interface ``jlev=0``       is top-of-atmosphere (TOA).
    * Interface ``jlev=nlev``    is the surface.
    * The op scaling for cloudy regions (op[1:] = op[0]**2 when op[0]>=0,
      else op[1:] = op[0]) matches the Fortran exactly.
    """
    frac_th = cloud_fraction_threshold
    nlev, nreg, nbatch = region_fracs.shape
    assert nreg == 3, "This implementation is hard-coded for nreg=3"
    assert overlap_param.shape == (nlev - 1, nbatch), (
        f"overlap_param must be (nlev-1={nlev-1}, nbatch={nbatch}), "
        f"got {overlap_param.shape}"
    )
 
    device = region_fracs.device
    dtype  = region_fracs.dtype
    nlev_p1 = nlev + 1
 
    v_matrix = torch.zeros(3, 3, nlev_p1, nbatch, device=device, dtype=dtype)
    zeros_b = torch.zeros(nbatch, device=device, dtype=dtype)
    zeros_rb = torch.zeros(3, nbatch, device=device, dtype=dtype)
    # TOA: upper layer = single clear-sky region
    frac_upper = torch.zeros(3, nbatch, device=device, dtype=dtype)
    frac_upper[0] = 1.0   # clear-sky region fraction = 1
 
    # op is irrelevant when only one region is present in the upper layer;
    # setting it to 1 reproduces the Fortran initialisation.
    op = torch.ones(3, nbatch, device=device, dtype=dtype)
 
    for jlev in range(nlev_p1):
 
        # ---- frac_lower: fraction of each region just below interface ----
        if jlev < nlev:
            frac_lower = region_fracs[jlev, :, :]      # (3, nbatch)
        else:
            # Surface: single clear-sky region
            frac_lower = zeros_rb
            frac_lower[0] = 1.0
 
        # ---- overlap parameter for this interface ----
        if jlev == 0 or jlev >= nlev:
            # TOA or surface: op irrelevant → set to 1
            op = torch.ones(3, nbatch, device=device, dtype=dtype)
        else:
            # Interior interface: index into overlap_param (0-based: jlev-1)
            op0 = overlap_param[jlev - 1]              # (nbatch,)
            op = torch.empty(3, nbatch, device=device, dtype=dtype)
            op[0] = op0
            # Cloudy regions: square when non-negative, same when negative
            op[1] = torch.where(op0 >= 0.0, op0 * op0, op0)
            op[2] = op[1]
 
        # ---- (3, 3, nbatch) non-directional overlap matrix ----
        # overlap_matrix = _calc_beta_overlap_matrix(
        #     op, frac_upper, frac_lower, frac_th
        # )
 
        oxm0 = op[0] * torch.minimum(frac_upper[0], frac_lower[0])  # (nbatch,)
        oxm1 = op[1] * torch.minimum(frac_upper[1], frac_lower[1])
        oxm2 = op[2] * torch.minimum(frac_upper[2], frac_lower[2])
    
        # --- denominator = 1 - sum(op_x_frac_min) ---
        denominator = 1.0 - (oxm0 + oxm1 + oxm2)                    # (nbatch,)
    
        # --- random part: factor * outer((frac_upper - oxm), (frac_lower - oxm)) ---
        valid = denominator >= frac_th                         # (nbatch,) bool
        factor = torch.where(valid, 1.0 / denominator.clamp(min=frac_th),
                            torch.zeros_like(denominator))           # (nbatch,)
    
        # residual fracs for upper and lower (3, nbatch)
        ru0 = frac_upper[0] - oxm0
        ru1 = frac_upper[1] - oxm1
        ru2 = frac_upper[2] - oxm2
    
        rl0 = frac_lower[0] - oxm0
        rl1 = frac_lower[1] - oxm1
        rl2 = frac_lower[2] - oxm2
    
        # Random part (9 elements, fully unrolled):
        # m[jupper, jlower] = factor * ru[jupper] * rl[jlower]
        # but only where denominator >= frac_th (factor is already zeroed)
        m00 = factor * ru0 * rl0
        m01 = factor * ru0 * rl1
        m02 = factor * ru0 * rl2
        m10 = factor * ru1 * rl0
        m11 = factor * ru1 * rl1
        m12 = factor * ru1 * rl2
        m20 = factor * ru2 * rl0
        m21 = factor * ru2 * rl1
        m22 = factor * ru2 * rl2
    
        # Add maximum part on diagonal:
        # overlap_matrix[jreg, jreg] += op_x_frac_min[jreg]
        m00 = m00 + oxm0
        m11 = m11 + oxm1
        m22 = m22 + oxm2
    
        # Stack into (3, 3, nbatch): first dim = jupper, second = jlower
        row0 = torch.stack([m00, m01, m02], dim=0)   # (3, nbatch)
        row1 = torch.stack([m10, m11, m12], dim=0)
        row2 = torch.stack([m20, m21, m22], dim=0)
        overlap_matrix = torch.stack([row0, row1, row2], dim=0)      # (3, 3, nbatch)
    

        # ---- convert to directional v_matrix and u_matrix ----
        # v_matrix[jlower, jupper, jlev] = overlap_matrix[jupper, jlower]
        #                                  / frac_upper[jupper]
        # u_matrix[jupper, jlower, jlev] = overlap_matrix[jupper, jlower]
        #                                  / frac_lower[jlower]
        #
        # Fully unrolled (jupper in 0..2, jlower in 0..2):
 
        # for jupper in range(3):
        #     fu = frac_upper[jupper]                    # (nbatch,) 
        #     valid_upper = fu >= frac_th                # (nbatch,) bool
        #     inv_fu = torch.where(valid_upper, 1.0 / fu.clamp(min=frac_th),
        #                          torch.zeros_like(fu))
 
        #     om_val = overlap_matrix[jupper, 0]  # (nbatch,)
        #     v_matrix[0, jupper, jlev] = torch.where(valid_upper, om_val * inv_fu, zeros_b)
        #     om_val = overlap_matrix[jupper, 1]  # (nbatch,)
        #     v_matrix[1, jupper, jlev] = torch.where(valid_upper, om_val * inv_fu, zeros_b)
        #     om_val = overlap_matrix[jupper, 2]  # (nbatch,)
        #     v_matrix[2, jupper, jlev] = torch.where(valid_upper, om_val * inv_fu, zeros_b)          
        # 
        jupper=0
        fu = frac_upper[jupper]                    # (nbatch,) 
        inv_fu = 1.0 / fu.clamp(min=frac_th)
        v_matrix[0, jupper, jlev] = overlap_matrix[jupper, 0] * inv_fu
        v_matrix[1, jupper, jlev] = overlap_matrix[jupper, 1] * inv_fu
        v_matrix[2, jupper, jlev] = overlap_matrix[jupper, 2] * inv_fu           
        jupper=1
        fu = frac_upper[jupper]                    # (nbatch,) 
        inv_fu = 1.0 / fu.clamp(min=frac_th)
        v_matrix[0, jupper, jlev] = overlap_matrix[jupper, 0] * inv_fu
        v_matrix[1, jupper, jlev] = overlap_matrix[jupper, 1] * inv_fu
        v_matrix[2, jupper, jlev] = overlap_matrix[jupper, 2] * inv_fu     
        jupper=2
        fu = frac_upper[jupper]                    # (nbatch,) 
        inv_fu = 1.0 / fu.clamp(min=frac_th)
        v_matrix[0, jupper, jlev] = overlap_matrix[jupper, 0] * inv_fu
        v_matrix[1, jupper, jlev] = overlap_matrix[jupper, 1] * inv_fu
        v_matrix[2, jupper, jlev] = overlap_matrix[jupper, 2] * inv_fu     
        # Slide the window down
        frac_upper = frac_lower
 
    return v_matrix

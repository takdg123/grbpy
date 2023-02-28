import numpy as np
from iminuit import Minuit
import matplotlib.pyplot as plt
from astropy.table import Table
from tqdm.notebook import trange
from copy import copy

import astropy.units as u

from astromodels.functions.function import (
    Function1D,
    FunctionMeta,
    ModelAssertionViolation,
)


class EFS_Model(Function1D, metaclass=FunctionMeta):
    r"""
    description :
        External forward shock model defined in J. Granot & R. Sari, 2002)
    latex : $ K~\left(0.5(\frac{x}{xb})^{low_index s}+0.5(\frac{x}{xb})^{high_index s}\right)^{(-1/s)} $
    parameters :
        K :
            desc : Normalization (photon flux at the cooling break)
            initial value : 1.0
            is_normalization : True
            transformation : log10
            unit : keV^-1cm^-2s^-1
            min : 1e-30
            max : 1e2
            delta : 0.1
        xb :
            desc : Cooling break
            initial value : 1
            unit : keV
            min: -2
            max: 2
            delta : 0.1
        low_index :
            desc : Low-energy spectral index
            initial value : 1.6
            min : 1.4
            max : 2
            delta : 0.01
        high_index :
            desc : High-energy spectral index
            initial value : 2.2
            min : 1.5
            max : 3
            delta : 0.01
    """

    def _set_units(self, x_unit, y_unit):
        self.low_index.unit = u.dimensionless_unscaled
        self.high_index.unit = u.dimensionless_unscaled

        self.xb.unit = u.keV
        self.K.unit = 1/u.cm**2/u.s/x_unit

    def evaluate(self, x, K, xb, low_index, high_index):
        xb = 10**xb
        xx = np.divide(x, xb)
        x1 = np.divide(1, xb)
        p = 2*(low_index-1)+1
        s = 0.80-0.03*p
        fnu = K*(xx**(s*low_index)+xx**(s*high_index))**(-1./s)
        fnu1 = (x1**(s*low_index)+x1**(s*high_index))**(-1./s)

        return fnu/fnu1


class EFS_Model_Cutoff(Function1D, metaclass=FunctionMeta):
    r"""
    description :
        External forward shock model defined in J. Granot & R. Sari, 2002) with a cutoff
    latex : $ K~\left(0.5(\frac{x}{xb})^{low_index s}+0.5(\frac{x}{xb})^{high_index s}\right)^{(-1/s)} \times exp(-x/xcutoff) $
    parameters :
        K :
            desc : Normalization (photon flux at the cooling break)
            initial value : 1.0
            is_normalization : True
            transformation : log10
            unit : keV^-1cm^-2s^-1
            min : 1e-30
            max : 1e2
            delta : 0.1
        xb :
            desc : Cooling break
            initial value : 1
            unit : keV
            min: -2
            max: 2
            delta : 0.1
        low_index :
            desc : Low-energy spectral index
            initial value : 1.6
            min : 1.4
            max : 2
            delta : 0.1
        low_index :
            desc : Low-energy spectral index
            initial value : 1.6
            min : 1.4
            max : 2
            delta : 0.1
        high_index :
            desc : High-energy spectral index
            initial value : 2.2
            min : 1.5
            max : 3
            delta : 0.1
        log10xc:
            desc : Exponential cutoff
            initial value : 7
            min: 4
            max: 12
            delta : 0.1
        
    """

    def _set_units(self, x_unit, y_unit):
        self.low_index.unit = u.dimensionless_unscaled
        #self.high_index.unit = u.dimensionless_unscaled

        self.xb.unit = u.keV
        
        self.K.unit = 1/u.cm**2/u.s/x_unit

    def evaluate(self, x, K, xb, low_index, high_index, log10xc):
        xb = 10**xb
        xx = np.divide(x, xb)
        x1 = np.divide(1, xb)
        p = 2*(low_index-1)+1
        s = 0.80-0.03*p
        #high_index = low_index+0.5
        fnu = K*(xx**(s*low_index)+xx**(s*high_index))**(-1./s)
        fnu1 = (x1**(s*low_index)+x1**(s*high_index))**(-1./s)

        xc = 10**log10xc
        cutoff = np.exp(-np.divide(x, xc))
        cutoff[cutoff < 1e-30] = 1e-30
        
        return fnu/fnu1*cutoff


# def shuffle_points(flux):
#     shuffled_f = []
#     flux = copy(flux)
#     for flx in flux.T:
#         np.random.shuffle(flx)
#         shuffled_f.append(flx)
#     return np.asarray(shuffled_f).T

# def run_simulation(energy, flux, fix_s=True, fix_ratio=False, cutoff=False, include_nuc=True, 
#                    p_init=[0.5, 5, 1, -1, 9], export=False, **kwargs):
#     if include_nuc:
#         if cutoff:
#             table = Table(names=["F", "m", "Eb", "m2", "s", "Ecutoff"])
#         else:
#             table = Table(names=["F", "m", "Eb", "m2", "s"])
#     else:
#         if cutoff:
#             table = Table(names=["F", "m", "Ecutoff"])
#         else:
#             table = Table(names=["F", "m"])
        
#     flux = shuffle_points(flux)
#     size = len(flux)
#     for i in trange(size):
#         l = efs_fit(energy, energy*flux[i], 
#                     fix_s=fix_s, fix_ratio=fix_ratio, cutoff=cutoff, include_nuc=include_nuc, 
#                     p_init = p_init)
#         l.fit(**kwargs)
#         table.add_row(l.p)

#     if export:
#         np.save(export, table)
#     else:
#         return table

# class efs_fit:
#     def __init__(self, energy=None, flux=None, medium="wind", fix_s=True, fix_ratio=False, 
#                  cutoff=True, include_nuc=True, p_init = [0.5, 3, 1, -1, 9]):

#         self.energy = energy
#         self.flux = flux
#         self.medium = medium

#         self.include_nuc = include_nuc
#         if not(include_nuc):
#             self.fix_ratio = True
#             self.fix_s = True
#         else:
#             self.fix_ratio = fix_ratio
#             self.fix_s = fix_s
            
#         self.p_init = p_init
#         self.cutoff = cutoff
        

#     def model_pl(self, E, F, m, **kwargs):

#         flx = 10**F*(E)**(-m)
#         return flx
    
#     def model(self, E, F, m, Eb, m2, s, **kwargs):
        
#         if kwargs.pop("fix_s", self.fix_s):
#             p = 2*m+1
#             if kwargs.pop("medium", self.medium) == "wind":
#                 s = 0.80-0.03*p
#             elif kwargs.pop("medium", self.medium) == "ism":
#                 s = 1.15-0.06*p
#         else:
#             s = 10**s
            
#         if kwargs.pop("fix_ratio", self.fix_ratio):
#             m2 = m+0.5
            
#         Eb = 10**Eb

#         flx = 10**F*((E/Eb)**(s*m)+(E/Eb)**(s*m2))**(-1./s)
#         return flx
    
#     def model_cutoff(self, E, F, m, Eb, m2, s, Ecutoff, **kwargs):
        
#         Ecutoff = 10**Ecutoff
        
#         if self.include_nuc:
#             flx = self.model(E, F, m, Eb, m2, s)*np.exp(-(E/Ecutoff))
#         else:
#             flx = self.model_pl(E, F, m)*np.exp(-E/Ecutoff-1)*np.exp(-(E/Ecutoff))
#         return flx

#     def likelihood(self, F, m, Eb, m2, s):
#         l = sum(abs(self.flux-self.model(self.energy, F, m, Eb, m2, s))/self.flux)
#         return l
    
#     def likelihood_pl(self, F, m):
#         l = sum(abs(self.flux-self.model_pl(self.energy, F, m))/self.flux)
#         return l

#     def likelihood_cutoff(self, F, m, Eb, m2, s, Ecutoff):
#         l = sum(abs(self.flux-self.model_cutoff(self.energy, F, m, Eb, m2, s, Ecutoff))/self.flux)
#         return l
    
#     def fit(self, **kwargs):
        
#         if self.cutoff:

#             minuit=Minuit(self.likelihood_cutoff, 
#                 F = np.log10(np.median(self.flux)), 
#                 m = self.p_init[0], 
#                 Eb = self.p_init[1],
#                 m2 = self.p_init[2],
#                 s = self.p_init[3],
#                 Ecutoff = self.p_init[4],
#                 )
#             minuit.limits["Ecutoff"] = (kwargs.pop("Ecutoff_min", 6), kwargs.pop("Ecutoff_max", 12))
#             minuit.fixed["Eb"] = not(self.include_nuc)

#         elif self.include_nuc:
#             minuit=Minuit(self.likelihood, 
#                 F = np.log10(np.median(self.flux)), 
#                 m = self.p_init[0], 
#                 Eb = self.p_init[1],
#                 m2 = self.p_init[2],
#                 s = self.p_init[3],
#                 )
#         else:
#             minuit=Minuit(self.likelihood_pl, 
#                 F = np.log10(np.median(self.flux)), 
#                 m = self.p_init[0], 
#                 )            

#         if self.include_nuc:
#             minuit.limits["s"] = (-3, 1)
#             minuit.limits["m2"] = (kwargs.pop("m2_min", 0.5), kwargs.pop("m2_max", 1.5))
#             minuit.limits["Eb"] = (kwargs.pop("Eb_min", 2), kwargs.pop("Eb_max", 7))

#             minuit.fixed["s"] = self.fix_s
#             minuit.fixed["m2"] = self.fix_ratio
            
#         minuit.limits["F"] = (-15, -10)
#         minuit.limits["m"] = (0, 1)

#         minuit.errordef=1
#         fit_result = minuit.migrad()

#         chisq = fit_result.fval
#         dof = len(self.flux) - len(fit_result.parameters)

#         self.p = fit_result.values
#         self.cov = fit_result.covariance
#         self.perr = fit_result.errors
#         self.stat = [chisq, dof]
#         self.minuit = minuit
#         self.fit_result = fit_result
#         self.valid = fit_result.valid
        
#     def viz(self, args):
#         plt.plot(self.energy, self.flux, "ok")
#         xm = np.geomspace(self.energy[0], self.energy[-1], 1000)
#         if self.cutoff:
#             plt.plot(xm, self.model_cutoff(xm, *args))
#         elif self.include_nuc:
#             plt.plot(xm, self.model(xm, *args))
#         else:
#             plt.plot(xm, self.model_pl(xm, *args))
#         plt.ylim(min(self.flux)/10, max(self.flux)*10)
#         plt.xscale("log")
#         plt.yscale("log")
#         plt.grid()
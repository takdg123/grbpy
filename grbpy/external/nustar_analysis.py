import numpy as np
import os
import glob

from astropy.table import Table
import astropy.units as u

from pathlib import Path
from IPython.display import clear_output

try:
    from xspec import AllData, AllModels, Xset, Model, Fit, Plot
except:
    print("xspec is not installed.")
    pass

from tqdm.notebook import tqdm

def read_qdp(file_name, t_shift=0):
    
    with open(file_name) as file:
        lines = file.readlines()
        test = np.array([line.find("flux") for line in lines])
        if len(test[test>0]) != 0:
            target="flux"
        else:
            target="gamma"
    
        vals = [line.split() for line in lines if (line[0] !="!") and (line[0] != " ") and (len(line.split()) ==6)]
    
    vals = np.asarray(vals).astype("float")

    tab = Table(vals, names=["time", "time_err_hi", "time_err_lo", f"{target}", f"{target}_err_hi", f"{target}_err_lo"])
    tab["time"] += t_shift

    return tab

def read_flux_table(file_name, t_shift=0):
    table = Table(np.load(file_name, allow_pickle=True))
    table["time"] += t_shift
    return table


def plot_lc(tab, ax=None, t_shift = 0, target="flux", **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    t = tab["time"]+t_shift
    if "time_err_lo" in tab.dtype.names:
        t_lo = tab["time_err_lo"]
        t_hi = tab["time_err_hi"]
    else:
        t_lo = tab["time_lo"]-tab["time"]
        t_hi = tab["time_hi"]-tab["time"]

    p = tab[f"{target}"]
    if f"{target}_err_lo" in tab.dtype.names:
        p_lo = tab[f"{target}_err_lo"]
        p_hi = tab[f"{target}_err_hi"]
    else:
        p_lo = tab[f"{target}_lo"]-tab[target]
        p_hi = tab[f"{target}_hi"]-tab[target]

    
    flag = (p_hi>0)*(p_lo<0)
    if target == "nH":
        flag = p<10

    ax.errorbar(t[flag], p[flag], 
                 xerr = [-t_lo[flag], t_hi[flag]], 
                 yerr = [-p_lo[flag], p_hi[flag]], ls="", **kwargs)
    ax.set_xscale("log")
    ax.set_xlabel("Time [s]")
    ax.grid(which="major")
    ax.grid(which="minor", ls=":", alpha=0.5)
    if target == "flux":
        ax.set_ylabel(r"Flux [erg/cm$^2$/s]")
        ax.set_yscale("log")
    elif target == "gamma":
        ax.set_ylabel("Photon index")


def calculate_flux(table, energy_band = [0.3, 10], units="keV", export=False):
    output = Table(names = ["time", "time_err_lo", "time_err_hi", "flux", "flux_err", "flux_err_lo", "flux_err_hi"])
    for i, tab in enumerate(table):
        mean = [tab["index"],tab["k"], tab["nH"]]
        params_sample = np.random.multivariate_normal(np.asarray(mean), 
                                               tab["cov"], 10000)
        idx = -params_sample[:,0]+2
        F = params_sample[:,1]/(idx)*(energy_band[1]**idx-energy_band[0]**idx)
        
        if np.percentile(F, 50) < 1e3:

            if units == "erg":
                F *= u.keV.to(u.erg)

            F = [tab["time"], tab["time_lo"]-tab["time"], tab["time_hi"]-tab["time"], 
                np.percentile(F, 50), np.std(F), np.percentile(F, 16)-np.percentile(F, 50), np.percentile(F, 84)-np.percentile(F, 50)]
            output.add_row(F)
    if export:
        np.save(export, output)
    else:
        return output

def get_butterfly(table, show_plot=False, ax=None, emin=0.3, emax = 10, scale=1, units="keV", **kwargs):

    E = np.geomspace(emin, emax, kwargs.pop("nbins", 101))/scale

    mean = [table["index"], table["k"], table["nH"]]
    params_sample = np.random.multivariate_normal(np.asarray(mean), 
                                           table["cov"], kwargs.pop("size", 10000))

    F = mean[1]*(E)**(-mean[0])/scale**2
    F_sample = np.asarray([params_sample[:,1]*(e)**(-params_sample[:,0])/scale**2 for e in E])

    if units == "erg":
        F *= u.keV.to(u.erg)
        F_sample *= u.keV.to(u.erg)

    F_band = np.asarray([[e, f, np.percentile(fs, 16), np.percentile(fs, 84)] for e, f, fs in zip(E, F, F_sample)])
    E = E*scale

    if show_plot:
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        
        props = ax.plot(E, E**2*F_band[:,1], **kwargs)
        ax.fill_between(E, E**2*F_band[:,2], E**2*F_band[:,3], 
                         color=props[0].get_color(), alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        return ax
    else:
        return F_band, E, F_sample.T

class xrt_analysis:
    

    fit_table = Table(dtype=[("file", str), ("index", float), ("index_lo", float), ("index_hi", float), 
                             ("k", float), ("k_lo", float), ("k_hi", float), 
                             ("nH", float), ("nH_lo", float), ("nH_hi", float), ("cov", list), (r"cstat", float), ("dof", int)])

    def __init__(self, data, mode = "pc", nH = 1e22, z=0, verbose=True):

        self.z = z
        self.nH = nH
        self.data = np.atleast_1d(data)
        self.mode = np.atleast_1d(mode)
        self.verbose = verbose
        
        
    def run_fit(self, **kwargs):
        if len(self.data) == len(self.mode):
            for d, m in tqdm(zip(self.data, self.mode), total=len(self.data)):
                self.fit_data(d, mode=m, **kwargs)
        elif len(self.mode) == 1:
            for d in tqdm(self.data):
                self.fit_data(d, mode=self.mode[0], **kwargs)
        
    def fit_data(self, data_path, mode="pc", **kwargs):
        data_path = str(Path(data_path).absolute())
        current_path = str(Path('.').absolute())
        AllData.clear()
        AllModels.clear()

        os.chdir(data_path)
        filename = glob.glob(f"*{mode}.pi")[0]
        AllData(filename)
        AllData(1).ignore("**-{}".format(kwargs.pop("emin", 0.3)))
        AllData(1).ignore("{}-**".format(kwargs.pop("emax", 10.)))
        os.chdir(current_path)

        models = 'powerlaw * TBabs * zTBabs'
        Mmanager=Model(models) 
        Mmanager.setPars({3: kwargs.pop("nH", self.nH)/1e22})
        Mmanager.setPars({5: kwargs.pop("z", self.z)})

        if kwargs.get("z_nH", False):
            Mmanager.setPars({4: kwargs.pop("z_nH")/1e22})
            Mmanager.zTBabs.nH.frozen=True

        Mmanager.zTBabs.Redshift.frozen=True
        Mmanager.TBabs.nH.frozen=True

        Xset.abund = "wilm"
        Xset.xsect = "vern"
        Xset.parallel.error = Mmanager.nParameters
        Xset.allowPrompting=False

        Fit.statMethod = "cstat"
        Fit.nIterations = 1000
        Fit.query = "no"
        Fit.perform()
        
        for i in range(Mmanager.nParameters):
            Fit.error('1.0 {}'.format(i+1))

        cov = self.reshape_cov(Fit.covariance)

        pars = self.parse_parameters(Mmanager)
        pars = [filename] + pars + [cov]+ [Fit.statistic, Fit.dof]
        self.fit_table.add_row(pars)
        
        self.Fit = Fit
        self.Mmanager = Mmanager
        if not(self.verbose):
            clear_output()

    def parse_parameters(self, m):
        par_list = []
        for par in ["PhoIndex", "norm"]:
            val = getattr(m.powerlaw, par)
            par_list+=[val.values[0]]+list(val.error[:2])

        val = m.zTBabs.nH
        par_list+=[val.values[0]]+list(val.error[:2])
        return par_list

    def reshape_cov(self, cov):
        temp=0
        for matSize in range(20):
            temp+=matSize
            if np.size(cov) == temp:
                break

        covReshape = np.zeros(shape=(matSize,matSize))

        k=0
        for j in range(matSize):
            for i in range(matSize):
                if i <= j and covReshape[i][j] == 0: 
                    covReshape[i][j]=cov[k]
                    covReshape[j][i]=cov[k]
                    k+=1
                    continue
                continue

        return covReshape

    def export(self, name="xrt_fit_result"):
        np.save(name, self.fit_table)
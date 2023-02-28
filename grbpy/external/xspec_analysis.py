import numpy as np
import os
from glob import glob

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
import matplotlib.pyplot as plt

empty_fit_table = Table(dtype=[("file", str), ("index", float), ("index_lo", float), ("index_hi", float), 
                             ("k", float), ("k_lo", float), ("k_hi", float), 
                             ("nH", float), ("nH_lo", float), ("nH_hi", float), ("cov", list), (r"cstat", float), ("dof", int)])


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


def plot_lc(tab, ax=None, t_shift = 0, target="flux", factor=1, **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    t = tab["time"]+t_shift

    if "time_err_lo" in tab.dtype.names:
        t_lo = tab["time_err_lo"]
        t_hi = tab["time_err_hi"]
    elif "time_lo" in tab.dtype.names:
        t_lo = tab["time_lo"]-tab["time"]
        t_hi = tab["time_hi"]-tab["time"]
    elif "time_min" in tab.dtype.names:
        t_lo = tab["time_min"]-tab["time"]
        t_hi = tab["time_max"]-tab["time"]

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

    prop = ax.errorbar(t[flag], p[flag]*factor, 
                 xerr = [abs(t_lo[flag]), t_hi[flag]], 
                 yerr = [-p_lo[flag]*factor, p_hi[flag]*factor], ls="", **kwargs)
    ax.set_xscale("log")
    ax.set_xlabel("Time [s]")
    ax.grid(which="major")
    ax.grid(which="minor", ls=":", alpha=0.5)
    if target == "flux":
        ax.set_ylabel(r"Flux [erg/cm$^2$/s]")
        ax.set_yscale("log")
    elif target == "gamma":
        ax.set_ylabel("Photon index")

    return ax, prop

def calculate_flux(table, energy_band = [0.5, 10], units="keV", export=False):
    output = Table(names = ["time", "time_err_lo", "time_err_hi", "flux", "flux_err", "flux_err_lo", "flux_err_hi"])
    for i, tab in enumerate(table):
        
        if tab["nH"] == 0:
            mean = [tab["index"],tab["k"]]
        else:
            mean = [tab["index"],tab["k"], tab["nH"]]
        
        params_sample = np.random.multivariate_normal(np.asarray(mean), 
                                               tab["cov"], 10000)
        idx = -params_sample[:,0]+2
        F = params_sample[:,1]/(idx)*(energy_band[1]**idx-energy_band[0]**idx)
        
        if np.percentile(F, 50) < 1e3:

            if units == "erg":
                F *= u.keV.to(u.erg)

            time = []
            if "time_lo" in tab.keys():
                time = [tab["time"], tab["time_lo"]-tab["time"], tab["time_hi"]-tab["time"]]
            elif "time_min" in tab.keys():
                t = (float(tab["time_min"])+float(tab["time_max"]))/2.
                time = [t, float(tab["time_min"])-t, float(tab["time_max"])-t]

            F = time + [np.percentile(F, 50), np.std(F), np.percentile(F, 16)-np.percentile(F, 50), np.percentile(F, 84)-np.percentile(F, 50)]
            output.add_row(F)
    if export:
        np.save(export, output)
    else:
        return output

def get_butterfly(table, show_plot=False, ax=None, emin=0.5, emax = 10, scale=1, units="keV", **kwargs):

    E = np.geomspace(emin, emax, kwargs.pop("nbins", 101))/scale

    if table["nH"] == 0:
        mean = [table["index"], table["k"]]
    else:
        mean = [table["index"], table["k"], table["nH"]]
    params_sample = np.random.multivariate_normal(np.asarray(mean), 
                                           table["cov"], kwargs.pop("size", 10000))

    const = kwargs.pop("const", 1)

    F = mean[1]*(E)**(-mean[0])/scale**2*const
    F_sample = np.asarray([params_sample[:,1]*(e)**(-params_sample[:,0])/scale**2*const for e in E])

    if units == "erg":
        F *= u.keV.to(u.erg)
        F_sample *= u.keV.to(u.erg)

    F_band = np.asarray([[e, f, np.percentile(fs, 16), np.percentile(fs, 84)] for e, f, fs in zip(E, F, F_sample)])
    E = E*scale

    if show_plot:
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        fill = kwargs.pop("fill", True)
        
        props = ax.plot(E, E**2*F_band[:,1], **kwargs)
        if fill:
            ax.fill_between(E, E**2*F_band[:,2], E**2*F_band[:,3], 
                             color=props[0].get_color(), alpha=0.3)
        else:
            ax.fill_between(E, E**2*F_band[:,2], E**2*F_band[:,3], 
                             facecolor="white", edgecolor=props[0].get_color(), alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        return ax
    else:
        return F_band, E, F_sample.T

def read_xrt_in_threeML(path, mode, name="XRT", **kwargs):

    from threeML.plugins.OGIPLike import OGIPLike

    p = Path(path)

    if kwargs.pop("rawdata", False):

        data = OGIPLike(
            name,
            observation=glob(str(p.absolute())+f"/sr.pha")[0],
            background=glob(str(p.absolute())+f"/bk.pha")[0],
            response=glob(str(p.absolute())+f"/sr.rmf")[0],
            arf_file=glob(str(p.absolute())+f"/sr.arf")[0],
        )

    else:

        data = OGIPLike(
            name,
            observation=glob(str(p.absolute())+f"/*{mode}source.pi")[0],
            background=glob(str(p.absolute())+f"/*{mode}back.pi")[0],
            response=glob(str(p.absolute())+f"/*{mode}.rmf")[0],
            arf_file=glob(str(p.absolute())+f"/*{mode}.arf")[0],
        )
    emin = kwargs.pop("emin", 0.5)
    emax = kwargs.pop("emax", 10.)
    data.set_active_measurements(f"{emin}-{emax}")
    
    if kwargs.pop("use_const", False):
        data.use_effective_area_correction()
    
    return data

def read_nustar_in_threeML(path, name="NuSTAR", teldef = "A", **kwargs):
    
    from threeML.plugins.OGIPLike import OGIPLike

    p = Path(path)

    data = OGIPLike(
        name+"_"+teldef,
        observation=glob(str(p.absolute())+f"/*{teldef}*sr.pha")[0],
        background=glob(str(p.absolute())+f"/*{teldef}*bk.pha")[0],
        response=glob(str(p.absolute())+f"/*{teldef}*sr.rmf")[0],
        arf_file=glob(str(p.absolute())+f"/*{teldef}*sr.arf")[0],
    )
    emin = kwargs.pop("emin", 3.)
    emax = kwargs.pop("emax", 79.)
    data.set_active_measurements(f"{emin}-{emax}")

    if kwargs.pop("use_const", False):
        data.use_effective_area_correction()
    
    return data

class xspec_analysis:
    
    
    XRT_E_BAND = [0.5, 10.]
    NUSTAR_E_BAND = [3., 79.]

    def __init__(self, data, instrument="XRT", mode = None, nH = 1e22, z=0, verbose=True):

        self.z = z
        self.nH = nH
        self.data = np.atleast_1d(data)
        self.instrument = instrument
        self.mode = np.atleast_1d(mode)
        self.verbose = verbose
        self.fit_table = empty_fit_table
        
        
    def run_fit(self, **kwargs):
        if len(self.data) == len(self.mode):
            for d, m in tqdm(zip(self.data, self.mode), total=len(self.data)):
                self.fit_data(d, m, **kwargs)
        elif len(self.mode) == 1:
            for d in tqdm(self.data):
                self.fit_data(d, self.mode[0], **kwargs)
        
    def fit_data(self, data_path, mode, **kwargs):
        data_path = str(Path(data_path).absolute())
        current_path = str(Path('.').absolute())
        AllData.clear()
        AllModels.clear()

        os.chdir(data_path)

        if self.instrument.lower() == "xrt":
            if kwargs.pop("rawdata", False) and os.path.exists("sr.pha"):
                filename = glob(f"sr.pha")[0]
                rsp_file = glob(f"sr.rmf")[0]
                arf_file = glob(f"sr.arf")[0]
                bk_file = glob(f"bk.pha")[0]
                AllData(filename)
                AllData(1).background = bk_file
                AllData(1).response = rsp_file
                AllData(1).response.arf = arf_file        
            else:
                filename = glob(f"*{mode}.pi")[0]
                AllData(filename)
            
        elif self.instrument.lower() == "nustar":
            filename = glob(f"*{mode}01_sr.pha")[0]
            rsp_file = glob(f"*{mode}01_sr.rmf")[0]
            arf_file = glob(f"*{mode}01_sr.arf")[0]
            bk_file = glob(f"*{mode}01_bk.pha")[0]
            AllData(filename)
            AllData(1).background = bk_file
            AllData(1).response = rsp_file
            AllData(1).response.arf = arf_file
            
        
        default_energy_band = getattr(self, f"{self.instrument.upper()}_E_BAND")
        AllData(1).ignore("**-{}".format(kwargs.pop("emin", default_energy_band[0])))
        AllData(1).ignore("{}-**".format(kwargs.pop("emax", default_energy_band[1])))
        os.chdir(current_path)

        models = 'powerlaw * TBabs '

        if self.instrument.lower() == "xrt":
            models += "* zTBabs"
        
        Mmanager=Model(models) 
        Mmanager.setPars({1: 1.8})
        Mmanager.setPars({3: kwargs.pop("nH", self.nH)/1e22})
        Mmanager.setPars({4: "1 0.01 0.01 0.001 5 10"})
        Mmanager.TBabs.nH.frozen=True

        if self.instrument.lower() == "xrt":
            if kwargs.get("z_nH", False):
                Mmanager.setPars({4: kwargs.pop("z_nH")/1e22})
                Mmanager.zTBabs.nH.frozen=True
            Mmanager.setPars({5: kwargs.pop("z", self.z)})
            Mmanager.zTBabs.Redshift.frozen=True
        

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

    def plot(self):
        Plot.commands = ()
        Plot.xLog = True
        Plot.yLog = True
        Plot.device = "/xs"
        Plot.xAxis="keV"
        
        Plot.setRebin(10, 100)
        Plot('data', 'residuals')
        Plot.iplot()
        
        self._Plot = Plot
        x = [Plot.x(), Plot.xErr()]
        y = [Plot.y(), Plot.yErr()]
        m = Plot.model()

        x = np.asarray(x).T
        y = np.asarray(y).T
        m = np.asarray(m)
        
        f, ax = plt.subplots(2,1, figsize=(5,4), gridspec_kw = {'height_ratios':[5, 1]}, sharex=True)
        plt.subplots_adjust(hspace=0.005)
        etc = ax[0].errorbar(x[:,0], y[:,0], xerr=x[:,1], yerr=y[:,1], ls="")
        ax[0].plot(x[:,0], m, color=etc[0].get_color())
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].grid()
        ax[1].errorbar(x[:,0], y[:,0]-m, xerr=x[:,1], yerr=y[:,1], ls="")
        ax[1].set_xscale("log")
        ax[1].grid()
        ax[0].set_ylabel("Counts")
        ax[1].set_ylabel("Data-Model")
        ax[1].set_xlabel("Energy [keV]")
        plt.show()


    def parse_parameters(self, m):
        par_list = []
        for par in ["PhoIndex", "norm"]:
            val = getattr(m.powerlaw, par)
            par_list+=[val.values[0]]+list(val.error[:2])

        if self.instrument.lower() == "xrt":
            val = m.zTBabs.nH
            par_list+=[val.values[0]]+list(val.error[:2])
        else:
            par_list+=[0, 0, 0]
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

    def export(self, name):
        np.save(name, self.fit_table)
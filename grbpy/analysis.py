import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from .utils import logger
from .plotting import fermi_plotter, plot_cnt_lc

from astropy import units as u
from astropy.coordinates import SkyCoord
try:
    from fermipy.gtanalysis import GTAnalysis
    from gt_apps import *
    from GtApp import GtApp
    gtsrcprob = GtApp('gtsrcprob')

except:
    print("Fermitools is not installed. Any Fermi-LAT related analysis cannot be performed.")

import fermipy.wcs_utils as wcs_utils
import fermipy.utils as fermi_utils

from regions import CircleSkyRegion
from pathlib import Path

from . import utils

from .utils import generatePHA, generatePSF, generateRSP

from .config import InitConfig

from astropy.table import Table

from IPython.display import Image, display
from glob import glob


def simple_analysis(name, info, tmin, tmax, verbosity=0, overwrite=False, **kwargs):
    config = InitConfig(outdir=f"{name}", 
                ra=info['ra'], 
                dec=info['dec'], 
                tmin=tmin+info["trigger"], 
                tmax=tmax+info["trigger"], 
                emin=info['emin'], 
                emax = info['emax'], 
                irf="SOURCE",
                datadir="data", 
                overwrite=overwrite, 
                target="GRB", 
                verbosity=verbosity)
    fix_all = kwargs.pop("fix_all", True)
    fix_galdiff = kwargs.pop("fix_galdiff", False)
    fermi = FermiAnalysis(config=config, 
                overwrite=overwrite, 
                trigger=info["trigger"], 
                grb_name=info["name"], 
                verbosity=verbosity, **kwargs)
    fermi.fit(fix_all=fix_all, fix_galdiff=fix_galdiff)
    fermi.analysis("sed", nbins=1)
    
    return fermi

def simple_load(config, info, table=None, **kwargs):
    if table is None:
        table = utils.lc_table()

    fermi = FermiAnalysis(config=f"{config}", 
                      trigger=info["trigger"], grb_name=info["name"], 
                      **kwargs)
    properties = [(fermi.basic_info["tmin"]+fermi.basic_info["tmax"])/2., 
                fermi.basic_info["tmin"], fermi.basic_info["tmax"]]

    sed = fermi.output["sed"]
    indices = []

    # for i, name in enumerate(sed["param_names"]):
    #     if name in [b"Integral", b"Index"]:
    #         properties += [ sed["param_values"][i], sed["param_errors"][i] ]
    #         indices.append(i)

    for key in fermi.fit_info.keys():
        properties += fermi.fit_info[key] 

    properties += [fermi.flux_info[key][0] for key in fermi.flux_info.keys()]
    properties += [fermi.output["fit"]["cov"]]
    
    table.add_row(properties)
    return table

class FermiAnalysis():
    """
    This is to perform a simple Fermi-LAT analysi. All fermipy.GTanalysis functions 
    and attributes can be accessed with the 'gta' arribute. e.g.,

        fermi = FermiAnalysis()
        fermi.gta.optimize()

    All information about the status of the analysis (See fermipy documentation
    for details) will be saved in the numpy array format (npy).

    Args:
        config (str or InitConfig): config for fermipy
            Default: config.yaml
        status_file (str): status filename (npy)
            Default: initial
        overwrite (bool): overwrite the status
            Default: False
        verbosity (int)
        **kwargs: passed to fermipy.GTAnalysis module
    """

    def __init__(self, config, status_file = "latest", target="GRB", overwrite=False, construct_dataset = False, verbosity=True, **kwargs):

        self._verbosity = verbosity
        self._logging = logger(self.verbosity)
        
        if type(config) is str:
            if ".yaml" in config:
                config = InitConfig.get_config(config)
            elif Path(config).is_dir():
                config = glob(str(Path(config).absolute())+"/*.yaml")[0]
                config = InitConfig.get_config(config)   
            else:
                try:
                    config = InitConfig.get_config(config+".yaml")   
                except:
                    self._logging.error("Check your config.")
        elif type(config) == InitConfig:
            config = config.info
        
        self._logging.info("Initialize the Fermi-LAT analysis.")

        target = config["selection"]["target"]
        config["selection"]["target"] = None
        self.trigger = kwargs.pop("trigger", None)
        self.grb_name = kwargs.pop("grb_name", None)

        self.gta = GTAnalysis(config, logging={'verbosity' : self.verbosity+1}, **kwargs)
        self.gta.config["selection"]["target"] = target
        self._outdir = self.gta.config['fileio']['outdir']

        nbins = kwargs.pop("nbins", 10)
        
        self._exist_rsp = False

        target_name = self.gta.config["selection"]["target"]

        self.output = {}

        if overwrite or not(os.path.isfile(f"{self._outdir}/{status_file}.fits")):
            
            if overwrite:
                self._logging.info("Overwrite the Fermi-LAT setup.")
            else:
                self._logging.info("Initial setup and configuration are not found. Performing the data reduction...")

            #os.system(f"rm -rf {self._outdir}/*")

            self._logging.debug("Generate fermipy files.")
            self.gta.setup(overwrite=overwrite)

            self.gta.config["data"]["ltcube"] = f"{self._outdir}/ltcube_00.fits"

            if target_name=="GRB" and not(self.gta.roi.has_source(target_name)):
                self.gta.add_source('GRB',{ 
                        'ra' : self.gta.config["selection"]["ra"], 
                        'dec' : self.gta.config["selection"]["dec"],
                        'SpectrumType' : 'PowerLaw2', 
                            'Index' : 2.0,
                            'Integral' : 1e-3, 
                            'LowerLimit': self.gta.config["selection"]["emin"],
                            'UpperLimit': self.gta.config["selection"]["emax"],
                        'SpatialModel' : 'PointSource' })
                self.set_target("GRB")
                self.gta.set_norm("galdiff", 1)
                self.gta.set_parameter("galdiff", "Index", 1.0)
                self.gta.free_parameter("galdiff", "Index", free=False)
                self.gta.free_norm("galdiff", free=False)

            self.gta.set_parameter_bounds("isodiff", "Normalization", [0.01, 10] )

            os.system(f"rm -rf {self._outdir}/*.png")
            self._logging.debug("Optimize the ROI.")
            

            self.gta.optimize()


            self.save_status(status_file, init=True, **kwargs)

            self._logging.info("The initial setup and configuration is saved [status_file = {}].".format(status_file))
        else:
            self._logging.info("The setup and configuration is found [status_file = {}]. Loading the configuration...".format(status_file))

            flag = self.load_status(status_file)

            if flag == -1:
                return

        
        self._test_model = {'Index' : 2.0, 'SpatialModel' : 'PointSource' }
        self._find_target()

        try:
            self._construct_event_table()
        except:
            self._event_table = None
            pass

        self._logging.info("Completed (Fermi-LAT initialization).")

    @property
    def target(self):
        """
        Return:
            fermipy.roi_model.Source
        """
        return self._target

    @property
    def target_name(self):
        """
        Return:
            str: target name
        """
        return self._target_name

    @property
    def target_id(self):
        """
        Return:
            int: target id
        """
        return self._target_id

    @property
    def event_table(self):
        return self._event_table

    @property
    def print_association(self):
        """
        Print sources within ROI and their associations.
        """

        for i, src in enumerate(self.gta.roi.sources):
            if src.name == "isodiff" or src.name=="galdiff":
                continue

            self._logging.info(str(i)+") "+src.name+":"+str(src.associations[1:]))

    @property
    def print_target(self):
        """
        Print the target properties
        """
        self._logging.info(self.gta.roi.sources[self.target_id])

    @property
    def print_model(self):
        """
        Print source models within ROI
        """
        return self.gta.print_model(loglevel=40)

    @property
    def print_params(self, full_output=False):
        """
        Print parameters of sources within ROI
        """
        return self.gta.print_params(False, loglevel=40)

    @property
    def print_info(self):
        config =  self.gta.config
        self._logging = logger()
        self._logging.info("-"*20+" Info "+"-"*20)
        self._logging.info("target: {}".format(config["selection"]["target"]))
        self._logging.info("localization:")
        self._logging.info("\t(ra, dec) : ({}, {})".format(config["selection"]["ra"],
                                            config["selection"]["dec"]))
        self._logging.info("time interval:")
        self._logging.info("\t{} - {}".format(utils.MET2UTC(config["selection"]["tmin"]),
                                          utils.MET2UTC(config["selection"]["tmax"])))
        self._logging.info("\tT_0+{:.2f} - T_0+{:.2f}".format(config["selection"]["tmin"]-self.trigger,
                                          config["selection"]["tmax"]-self.trigger))
        self._logging.info("energy band:")
        self._logging.info("\t{:.1f} - {:.1f} GeV".format(float(config["selection"]["emin"])/1e3, float(config["selection"]["emax"])/1e3))
        self._logging.info("-"*45)

    @property
    def verbosity(self):
        """
        Return:
            int
        """
        return self._verbosity

    @property
    def basic_info(self):
        return {
            "trigger": self.trigger,
            "tmin": self.gta.config["selection"]["tmin"]-self.trigger,
            "tmax": self.gta.config["selection"]["tmax"]-self.trigger,
            "emin": self.gta.config["selection"]["emin"],
            "emax": self.gta.config["selection"]["emax"]
        }
    
    @property
    def fit_info(self):
        f = self.output["fit"]
        scales = self.gta.roi[self.target_name].spectral_pars
        return {n: [f["values"][i]*scales[n]["scale"], abs(f["errors"][i]*scales[n]["scale"]), scales[n]["scale"]]  for i, n in enumerate(f["names"])}
        
    @property
    def flux_info(self):
        properties = ["ts", "e2dnde", "e2dnde_err", "e2dnde_err_hi", "e2dnde_err_lo", "e2dnde_ul95"]
        return {prop: self.output["sed"][prop] for prop in properties}

    def show_fit_result(self):
        pngs = glob(f'{self._outdir}/simple*.png')

        for png in pngs:
            my_image = Image(png)
            display(my_image)
                
    def src_prob(self, status = "simple"):

        if not(os.path.exists(self._outdir+f'/{status}_00.xml')):
            self._logging.error("Run FermiAnalysis.fit first.")
            return

        maketime['evfile'] = self._outdir+'/ft1_00.fits'
        maketime['outfile'] = self._outdir+'/ft1_filtered_00.fits'
        maketime['scfile'] = self.gta.config["data"]["scfile"]
        maketime['filter'] = '(DATA_QUAL>0||DATA_QUAL==-1||DATA_QUAL==1)&&(LAT_CONFIG==1)'
        maketime['apply_filter'] = 'yes'
        maketime['roicut'] = 'yes'
        maketime['overwrite'] = 'yes'
        maketime['chatter'] = int(self.verbosity)-1
        maketime.run()
        
        diffResps['evfile'] = self._outdir+'/ft1_filtered_00.fits'
        diffResps['scfile'] = self.gta.config["data"]["scfile"]
        diffResps['srcmdl'] = self._outdir+'/simple_00.xml'
        diffResps['irfs'] = self.gta.config["gtlike"]["irfs"]
        diffResps['evtype'] = self.gta.config["selection"]["evtype"]
        diffResps['chatter'] = int(self.verbosity)-1
        diffResps.run()

        gtsrcprob["evfile"] = self._outdir+'/ft1_filtered_00.fits'
        gtsrcprob["scfile"] = self.gta.config["data"]["scfile"]
        gtsrcprob["outfile"] = self._outdir+'/ft1_srcprob_00.fits'
        gtsrcprob["srcmdl"] = self._outdir+f'/{status}_00.xml'
        gtsrcprob["irfs"] = self.gta.config["gtlike"]["irfs"]
        gtsrcprob["evtype"] = self.gta.config["selection"]["evtype"]
        gtsrcprob['chatter'] = int(self.verbosity)-1
        gtsrcprob.run() 

        self._construct_event_table()

    def _construct_event_table(self):
        table = Table(fits.open(self._outdir+'/ft1_srcprob_00.fits')[1].data)
        table["TIME"] -= self.trigger 
        self._event_table  = table["ENERGY", "RA", "DEC", "TIME", "Source"]


    def peek_lc(self, grb_only=False, binsz=10):

        if self.event_table is None:
            try:
                self._construct_event_table()
            except:
                self.src_prob()

        if self.event_table is None:
            return

        if grb_only:
            event = self.event_table[self.event_table["Source"]>0.9]
        else:
            event = self.event_table
        event = self.event_table

        ax, temp = plot_cnt_lc(event, binsz=10, c=event["Source"])
        return ax

    def peek_irfs(self):
        """
        Show instrument response function (irf) information
        """
        if not(hasattr(self, "datasets")):
            self._logging.error("Run FermiAnalysis.construct_dataset first.")
            return

        edisp_kernel = self.datasets.edisp.get_edisp_kernel()

        f, ax = plt.subplots(2,2, figsize=(10, 6))
        edisp_kernel.plot_bias(ax = ax[0][0])
        ax[0][0].set_xlabel(f"$E_\\mathrm{{True}}$ [keV]")

        edisp_kernel.plot_matrix(ax = ax[0][1])
        self.datasets.psf.plot_containment_radius_vs_energy(ax = ax[1][0])
        self.datasets.psf.plot_psf_vs_rad(ax = ax[1][1])
        plt.tight_layout()

    def save_status(self, status_file, init=False):
        """
        Save the status

        Args:
            status_file (str): passed to fermipy.write_roi
            init (bool): check whether this is the initial analysis.
                Default: False
        """

        if (init==False) and (status_file == "initial"):
            self._logging.warning("The 'inital' status is overwritten. This is not recommended.")
            self._logging.warning("The original 'inital' status is archived in the '_orig' folder.")
            os.system(f"mkdir ./{self._outdir}/_orig")
            for file in glob.glob(f"{self._outdir}/*initial*"):
                os.sytem(f"mv {file} {self._outdir}/_orig/")

        self.gta.write_roi(status_file, save_model_map=True)
        self.gta.write_roi("latest", save_model_map=True)
        self._fermi_status = status_file

    def load_status(self, status_file):
        """
        Load the status

        Args:
            status_file (str): passed to fermipy.write_roi
        """
        filename = f"{self._outdir}/{status_file}.fits"
        if os.path.exists(filename):
            self.gta.load_roi(status_file)
            self._fermi_status = status_file
            filename = f"{self._outdir}/{status_file}_output.npy"
            if os.path.exists(filename):
                self.output = np.load(filename, allow_pickle=True).item()
            filename = f"{self._outdir}/gtrsp_00.rsp"
            if os.path.exists(filename):
                self._exist_rsp = True
            else:
                self._exist_rsp = False

        else:
            self._logging.error("The status file does not exist. Check the name again")
            return -1

    def set_target(self, target):
        """
        Set/change the target

        Args:
            target (str or int): target name or id
        """
        if type(target)==int:
            self._target = self.gta.roi.sources[target]
            self._target_id = target
            self._logging.info(f"The target is set to {self.gta.roi.sources[target].name}")
            return
        elif type(target)==str:
            self._find_target(name=target)
            self._logging.info(f"The target is set to {target}")

    def remove_weak_srcs(self, ts_cut=1, npred_cut=0):

        """
        Remove sources within ROI if they are too weak.
        Args:
            ts_cut (float): remove sources with a TS cut
                Default: 1
            npred_cut (float): remove sources with a npred cut
                Default: 0
        """
        N = 0
        for src in self.gta.roi.sources:
            if (src.name == "isodiff") or (src.name=="galdiff") or (src.name == self.target_name):
                continue

            if src.skydir.separation(self.target.skydir)<0.01*u.deg:
                continue

            if np.isnan(src['ts']) or src['ts'] < ts_cut or src['npred'] < npred_cut:
                self.gta.delete_source(src.name)
                N+=1
        self._logging.info(f"{N} sources are deleted.")


    def fit(self, status_file="simple", pre_status=None,
        optimizer = 'NEWMINUIT', fix_all=True,
        fix_galdiff=True, remove_weak_srcs=False,
        return_output=False, **kwargs):
        """
        Perform a simple fitting with various cuts

        Args:
            status_file (str): output status filename (npy)
                Default: simple
            optimizer (str): either MINUIT or NEWMINUIT
                Default: NEWMINUIT
            return_output (bool): return the fitting result (dict)
                Default: False
            pre_status (str, optional): input status filename (npy). If not defined, starting from
                the current status.
                Default: None
            **kwargs: passed to fermipy.GTAnalysis.free_sources function

        Return
            dict: the output of the fitting when return_output is True
        """
        if pre_status is not None:
            self.load_status(pre_status)

        if fix_all:
            self.gta.free_sources(free=False)
        else:
            self.gta.free_sources(free=False)

            self.gta.free_sources(free=True, distance=kwargs.pop("distance", 3.0),  pars='norm', **kwargs)

            self.gta.free_sources(free=True, minmax_ts=[kwargs.pop("min_ts", 5), None], **kwargs)

        self.gta.free_sources_by_name(self.target_name, free=True, pars=None)
        self.gta.free_sources_by_name("isodiff", free=True, pars=None)

        if fix_galdiff:
            self.gta.set_norm("galdiff", 1)
            self.gta.free_norm("galdiff", free=False)
            self.gta.free_parameter("galdiff", "Index", free=False)
        else:
            self.gta.set_norm("galdiff", 1)
            self.gta.free_norm("galdiff", free=True)
            self.gta.free_parameter("galdiff", "Index", free=False)

        self._fit_result = self.gta.fit(optimizer=optimizer, reoptimize=True, min_fit_quality=2, verbosity=False)
        
        if remove_weak_srcs:
            self.remove_weak_srcs()
            self._fit_result = self.gta.fit(optimizer=optimizer, reoptimize=True, min_fit_quality=2, verbosity=False)

        if self._fit_result["fit_success"]:
            self._logging.info("Fit successfully ({}).".format(self._fit_result["fit_quality"]))
        else:
            self._logging.error("Fit failed.")

        self.save_status(status_file)

        self._logging.info(f"The status is saved as '{status_file}'. You can load the status by vtspy.FermiAnalysis('{status_file}').")
        
        indices = [i for i, name in enumerate(self._fit_result["src_names"]) if name == self.target_name]
        names = np.array(self._fit_result["par_names"])[indices]
        values = self._fit_result["values"][indices]
        errors = self._fit_result["errors"][indices]
        cov = self._fit_result["covariance"][indices][:,indices]
        
        self.output["fit"] = {"names": names, "values": values, "errors": errors, "cov": cov}

        np.save(f"{self._outdir}/{status_file}_output", self.output)
        np.save(f"{self._outdir}/latest_output", self.output)
        if return_output:
            return self._fit_result

    def analysis(self, jobs = ["ts", "resid", "sed"], status_file="analyzed", **kwargs):
        """
        Perform various analyses: TS map, Residual map, and SED.

        Args:
            jobs (str or list): list of jobs, 'ts', 'resid', and/or 'sed'.
                Default: ['ts', 'resid', 'sed']
            status_file (str): output status filename (npy)
                Default: analyzed
            **kwargs: passed to GTanalysis.sed
        """

        model = kwargs.get("model", self._test_model)
        free = self.gta.get_free_param_vector()

        if type(jobs) == str:
            jobs = [jobs]

        if "ts" in jobs:
            o = self._ts_map(model=model)
            self.output['ts'] = o

        if "resid" in jobs:
            o = self._resid_dist(model=model)
            self.output['resid'] = o

        if "sed" in jobs:
            outfile=status_file+"_sed.fits"
            o = self._calc_sed(outfile=outfile, **kwargs)
            self.output['sed'] = o

        self.gta.set_free_param_vector(free)

        np.save(f"{self._outdir}/{status_file}_output", self.output)
        np.save(f"{self._outdir}/latest_output", self.output)

        self.save_status(status_file)
        self._logging.info(f"The status is saved as '{status_file}'. You can load the status by vtspy.FermiAnalysis('{status_file}').")

    def plot(self, output, status_file="analyzed", **kwargs):
        """
        Show various plots: TS map, Residual map, and SED.

        Args:
            output (str or list): list of plots to show
                Options: ["sqrt_ts", "npred", "ts_hist", "data",
                "model", "sigma", "excess", "resid", "sed"]
            status_file (str): read the output (from FermiAnalysis.analysis)
        """

        if not(hasattr(self, "output")):
            filename = f"{self._outdir}/{status_file}_output.npy"
            if os.path.exists(filename):
                self._logging.info("Loading the output file...")
                self.output = np.load(filename, allow_pickle=True).item()
            else:
                self._logging.error("Run FermiAnalysis.analysis first.")
                return

        list_of_fig = ["sqrt_ts", "npred", "ts_hist",
                        "data", "model", "sigma", "excess", "resid",
                        "sed"]

        if type(output) == str:
            if output == "ts":
                output = ["sqrt_ts", "ts_hist"]
            else:
                output = [output]

        for o in output:
            if o not in list_of_fig:
                output.remove(o)

        if len(output) == 1:
            sub = "11"
        elif len(output) <= 3:
            sub = "1"+str(len(output))
            f = plt.figure(figsize=(4*len(output), 4))
        elif len(output) == 4:
            sub = "22"
            f = plt.figure(figsize=(8, 8))
        elif len(output) == 6:
            sub = "23"
            f = plt.figure(figsize=(12, 8))

        for i, o in enumerate(output):
            subplot = int(sub+f"{i+1}")
            ax = fermi_plotter(o, self, subplot=subplot, **kwargs)
            
        plt.tight_layout()
        plt.show()

    def find_sources(self, status_file = "wt_new_srcs", re_fit=True, return_srcs=False, **kwargs):
        """
        Find sources within the ROI (using GTanalysis.find_sources).

        Args:
            status_file (str): output status filename (npy)
                Default: wt_new_srcs
            re_fit (bool): re fit the ROI with new sources
                Default: True
            return_srcs (bool): return source dictionaries
                Default: False
            **kwargs: passed to fit.

        Return:
            dict: output of GTanalysis.find_sources
        """

        self.gta.set_norm("isodiff", 1)
        
        srcs = self.gta.find_sources(model=self._test_model, sqrt_ts_threshold=5.0,
                        min_separation=0.5)

        self._logging.info("{} sources are found. They are added into the model list.".format(len(srcs["sources"])))

        if re_fit:
            self.fit(status_file=status_file, **kwargs)
        else:
            self.save_status(status_file)

        if return_srcs:
            return srcs

    def remove_weak_srcs(self, ts_cut=0, npred_cut=0):

        """
        Remove sources within ROI if they are too weak.
        Args:
            ts_cut (float): remove sources with a TS cut
                Default: 0
            npred_cut (float): remove sources with a npred cut
                Default: 0
        """
        N = 0
        for src in self.gta.roi.sources:
            if (src.name == "isodiff") or (src.name=="galdiff") or (src.name == self.target_name):
                continue

            if src.skydir.separation(self.target.skydir)<0.01*u.deg:
                continue

            if np.isnan(src['ts']) or src['ts'] < ts_cut or src['npred'] < npred_cut:
                self.gta.delete_source(src.name)
                N+=1
        self._logging.info(f"{N} sources are deleted.")

    def _find_target(self, name=None):
        if name is None:
            name = self.gta.config['selection']['target']

        flag = False
        for i, src in enumerate(self.gta.roi.sources):
            if src.name == "isodiff" or src.name=="galdiff":
                continue

            for n in src.associations:
                if (n.replace(" ", "") == name) or (n == name):
                    self._target = self.gta.roi.sources[i]
                    self._target_name = self.gta.roi.sources[i].name
                    self._target_id = i
                    list_of_association = src.associations
                    flag = True
                    self.gta.config["selection"]["target"] = self.target_name
            if flag:
                break

        if flag:
            self._logging.debug("The target, {}, is associated with {} source(s).".format(self.target_name, len(list_of_association)-1))
            self._logging.debug(list_of_association)
        else:
            self._logging.warning("The target name defined in the config file is not found.")
            self._target = self.gta.roi.sources[0]
            self._target_name = self.target.name
            self._target_id = 0

    def _ts_map(self, model):
        self._logging.info("Generating a TS map...")
        self.gta.free_sources(free=False)
        self.gta.free_sources(pars="norm")
        o = self.gta.tsmap('ts', model=model, write_fits=True, write_npy=True, make_plots=True)
        self._logging.info("Generating the TS map is completed.")
        return o

    def _resid_dist(self, model):
        self._logging.info("Generating a residual distribution...")
        self.gta.free_sources(free=False)
        self.gta.free_sources(pars="norm")
        o = self.gta.residmap('resid',model=model, write_fits=True, write_npy=True)
        self._logging.info("Generating the residual distribution is completed.")
        return o

    def _calc_sed(self, target=None, outfile = 'sed.fits', **kwargs):
        self._logging.info("Generating a SED... ")

        if target is None:
            target = self.target_name

        energy_bounds = kwargs.pop("energy_bounds", [self.gta.config["selection"]["emin"], self.gta.config["selection"]["emax"]])
        
        loge_bins = np.linspace(np.log10(energy_bounds[0]), np.log10(energy_bounds[1]), kwargs.pop("nbins", 10)+1)
        
        if len(loge_bins) == 2:
            use_local_index = True
        else:
            use_local_index = False

        o = self.gta.sed(self.target.name, outfile=outfile, use_local_index=use_local_index,
            bin_index=kwargs.pop("bin_index", 2.0), loge_bins=loge_bins, write_fits=True, write_npy=True, **kwargs)
        self._logging.info("Generating the SED is completed.")
        return o


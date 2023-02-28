import os
import glob

import yaml

from astropy.io import fits

from . import utils
from .utils import logger
from .const import SCRIPT_DIR

from pathlib import Path

import numpy as np


class InitConfig:
	"""
	This is to generate the configuration file compatible to
	the Fermipy configuration. 

	Args:
		outdir (str): a path to the output directory
	    		Default: lat
	    file_name (str): Fermi config filename (yaml)
	    	Default: config.yaml
	    verbosity (int)
	    **kwargs: passed to JointConfig.init
	"""

	def __init__(self, file_name="config.yaml", outdir = "./", verbosity=1, overwrite=True, **kwargs):
		
		self._logging = logger(verbosity=verbosity)

		if ".yaml" not in file_name:
			file_name = file_name+".yaml"

		self._filename = file_name

		basedir = kwargs.pop("basedir", "./")
		basedir = str(Path(basedir).absolute())
		self._outdir = Path(basedir, outdir)
		self._path = Path(self._outdir, file_name)

		if self.path.is_file():
			self.info = self.get_config(self.path)
			if verbosity:
				self.print_config(self.path)
			self._logging.info(f'a configuration file ({str(self.path.absolute())}) is loaded.')
		else:
			self.info = self._empty4fermi(**kwargs)
			self.set_config(self.path, self.info)
			if verbosity:
				self.print_config(self.path)
			self._logging.info(f'a configuration file ({str(self.path.absolute())}) is created.')

	
	@property
	def path(self):
		return self._path

	def change_time_interval(self, tmin, tmax, scale = "utc", instrument="all"):
		"""
		Change and update a time interval

		Args:
		tmin (float or str): start time
		tmax (float or str): end time
		scale (str): "utc", "mjd", or "met"
			Default: "utc"
		instrument (str): "fermi", "veritas", or "all"
			Default: "all"


		"""
		if scale.lower() == "utc":
			tmin_mjd = utils.UTC2MJD(tmin)
			tmax_mjd = utils.UTC2MJD(tmax)
			tmin_met= utils.UTC2MET(tmin[:10])
			tmax_met = utils.UTC2MET(tmax[:10])+60*60*24
		elif scale.lower() == "mjd":
			tmin = float(tmin)
			tmax = float(tmax)
			tmin_mjd = tmin
			tmax_mjd = tmax
			tmin_utc = utils.MJD2UTC(tmin)
			tmax_utc = utils.MJD2UTC(tmax)
			tmin_met = utils.UTC2MET(tmin_utc[:10])
			tmax_met = utils.UTC2MET(tmax_utc[:10])+60*60*24
		elif scale.lower() == "met":
			tmin = float(tmin)
			tmax = float(tmax)
			tmin_met = tmin
			tmax_met = tmax
			tmin_utc = utils.MET2UTC(tmin)
			tmax_utc = utils.MET2UTC(tmax)
			tmin_mjd = utils.UTC2MJD(tmin_utc)
			tmax_mjd = utils.UTC2MJD(tmax_utc)
		else:
			self._logging.error("The input 'scale' parameter is not 'MJD', 'MET', or 'UTC'.")
			return

		if instrument.lower() == "fermi" or instrument.lower() == "all":
			self.config["selection"]["tmin"] = tmin_met
			self.config["selection"]["tmax"] = tmax_met

		
		self.set_config(self.path, self.config)
		self.print_info(self.path)


	@staticmethod
	def get_config(path):
		"""
	    Read a config file.

	    Args:
	    	path (str): a path to a config file
		"""
		if ".yaml" not in str(path):
			path = str(path)+".yaml"


		return yaml.load(open(path), Loader=yaml.FullLoader)

	def set_config(self, path, info):
		"""
	    Write inputs into a config file.

	    Args:
	    	path (str): a path to a config file
	    	info (dict): overwrite the input info into a config file
		"""
		if ".yaml" not in str(path):
			path = str(path)+".yaml"

		with open(path, "w") as f:
			yaml.dump(info, f)


	@classmethod
	def print_config(self, path):
		self.config = self.get_config(path)
		
		self._logging = logger()
		self._logging.info("-"*20+" Info "+"-"*20)
		self._logging.info("target: {}".format(self.config["selection"]["target"]))
		self._logging.info("localization:")
		self._logging.info("\t(ra, dec) : ({}, {})".format(self.config["selection"]["ra"],
		                                    self.config["selection"]["dec"]))
		self._logging.info("\t(glat, glon) : ({}, {})".format(self.config["selection"]["glat"],
		                                    self.config["selection"]["glon"]))
		self._logging.info("time interval:")
		self._logging.info("\t{} - {}".format(utils.MET2UTC(self.config["selection"]["tmin"]),
		                                  utils.MET2UTC(self.config["selection"]["tmax"])))
		self._logging.info("-"*45)


	@staticmethod
	def _filter(pre_info, info):
		if len(info) != 0:
			for key in list(info.keys()):
				for subkey in list(info[key].keys()):
					if (pre_info[key][subkey] == info[key][subkey]) or (info[key][subkey]==None):
						info[key].pop(subkey)
		return info

	def _empty4fermi(self, gald = "gll_iem_v07.fits", irf="TRANSIENT020E", **kwargs):
		
		self._outdir.mkdir(parents=True, exist_ok=True)
		
		datadir = kwargs.pop("datadir", None)

		if datadir is not None:
			datadir = Path(str(self._outdir.parent), datadir)
			datadir.mkdir(parents=True, exist_ok=True)
		else:
			datadir = self._outdir

		if irf == "TRANSIENT020E":
			evclass = 8
		elif irf == "SOURCE":
			evclass = 128
		
		info = {
 				'data': {
 					'evfile': f"{datadir}/EV00.lst",
 					'scfile': f"{datadir}/SC00.fits",
 					'ltcube': None
 					},
 				'binning': {
 					'roiwidth': 12,
  					'binsz': 0.2,
  					'binsperdec': 10,
  					'coordsys': "CEL",
  					'projtype': 'WCS',
  					},
 				'selection': {
 					'radius': 12,
 					'emin': 100,
					'emax': 100000,
					'tmin': None,
					'tmax': None,
					'zmax': 100,
					'evclass': evclass,
					'evtype': 3,
					'glon': None,
					'glat': None,
					'ra': None,
					'dec': None,
					'target': None,
					'filter': "(DATA_QUAL>0||DATA_QUAL==-1||DATA_QUAL==1)&&(LAT_CONFIG==1)",
					'roicut': 'yes'
					},
				'gtlike': {
					'edisp': True,
					'irfs': f'P8R3_{irf}_V3',
					'edisp_disable': ['isodiff', 'galdiff']
					},
				'model': {
					'src_radius': 12,
					'galdiff': f'$FERMI_DIFFUSE_DIR/{gald}',
					'isodiff': f'$FERMI_DIFFUSE_DIR/iso_P8R3_{irf}_V3_v1.txt',
					'catalogs': ['4FGL-DR3'],
					
					},
				'fileio': {
					'outdir' : f"{self._outdir}",
					'usescratch': False,
					},
				}


		for key in info:
		    for val in info[key]:
		        info[key][val] = kwargs.pop(val, info[key][val])
		return info

	
	def create_threeml_config(self):
		from threeML import FermipyLike

		threeml_config = FermipyLike.get_basic_config(
		    evfile=self.info["data"]["evfile"],
		    scfile=self.info["data"]["scfile"],
		    ra = self.info["selection"]["ra"],
		    dec = self.info["selection"]["dec"],
		)

		for key in list(self.info.keys()):
			for subkey in list(self.info[key].keys()):
				if key in threeml_config.keys():
					if subkey in threeml_config[key].keys():
						threeml_config[key][subkey] = self.info[key][subkey]

		return threeml_config
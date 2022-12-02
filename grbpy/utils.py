import os
import numpy as np
import copy
import glob
from pathlib import Path

from astropy.time import Time, TimeDelta
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

import logging

logging.basicConfig(format=('%(asctime)s %(levelname)-8s: %(message)s'), datefmt='%Y-%m-%d %H:%M:%S', level=20)

SCRIPT_DIR = str(Path(__file__).parent.absolute())

MeV2Erg = 1.60218e-6
TeV2Erg = 1.60218

def lc_table(**kwargs):
    return Table(names = ["time", "t_min", "t_max", 
                    "par_1", "par_1_err", "par_2", "par_2_err", 
                    "ts", "e2dnde", "e2dnde_err", 
                    "e2dnde_err_hi", "e2dnde_err_lo", "e2dnde_ul95"],
                    **kwargs)

def logger(verbosity = 1):
    """
    Set a log level:

    * 1: info, warning, error
    * 2: debug, info, warning, error
    * 0: warning, error
    * -1: error

    Args:
        verbosity (int)
            Default: 1
    
    Return:
        astorpy.time: UTC time
    """
    levels_dict = {2: 10,
                   1: 20,
                   0: 30,
                   -1: 40,
                   -2: 50}
                   
    level = levels_dict[verbosity]
    logging.getLogger().setLevel(level)
    return logging

def MET2UTC(met, return_astropy=False):
    """
    Convert Fermi MET (Mission Elapsed Time) to UTC (Coordinated 
    Universal Time).

    Args:
        met (float): MET time in seconds
        return_astropy (bool): return astropy.time
    
    Return:
        str, astropy.time (optional): UTC time
    """
    if met is None:
        return None

    refMET = Time('2001-01-01', format='isot', scale='utc')

    dt = TimeDelta(met, format='sec')

    if return_astropy:
        return (refMET+dt).iso, refMET+dt
    else:
        return (refMET+dt).iso

def UTC2MET(utc):
    """
    Convert UTC to Fermi MET (mission elapsed time).

    Args:
        utc (astorpy.time): UTC time
    
    Return:
        float: MET
    """
    if utc is None:
        return None
        
    refMET = Time('2001-01-01', format='isot', scale='utc')
    currentTime = Time(utc, format='isot', scale='utc')
    return float((currentTime-refMET).sec)

def METnow():
	"""
    Return the current MET.

    Return:
        float: Fermi MET
    """
	refMET = Time('2001-01-01', format='isot', scale='utc')
	currentTime = Time.now()
	return float((currentTime-refMET).sec)

def MJD2UTC(mjd, return_astropy=False):
    """
    Convert MJD (Modified Julian Day) to UTC.

    Args:
        mjd (astorpy.time): MJD time
        return_astropy (bool): return astropy.time

    Return:
        str, astropy.time (optional): UTC time
    """
    if mjd is None:
        return None

    refMJD = Time(mjd, format='mjd')
    if return_astropy:
        return refMJD.isot, refMJD
    else:
        return refMJD.isot

def UTC2MJD(utc):
    """
    Convert UTC to MJD.

    Args:
        mjd (astorpy.time): MJD time
    
    Return:
        float: MJD
    """

    if utc is None:
        return None
        
    refUTC= Time(utc, format='isot', scale='utc')
    return float(refUTC.mjd)

def CEL2GAL(ra, dec):
    """
    Convert CEL (celestial) coordinates to GAL (galactic) coordinates.

    Args:
        ra (float): right ascension in degrees
        dec (float): declination in degrees

    Return:
        deg, deg
    """
    if ra is None or dec is None:
        return None, None

    c = SkyCoord(ra=float(ra)*u.degree, dec=float(dec)*u.degree, frame='icrs')
    return c.galactic.l.deg, c.galactic.b.deg

def GAL2CEL(l, b):
    """
    Convert MJD (Modified Julian Day) to UTC.

    Args:
    mjd (astorpy.time): MJD time

    Return:
        astropy.time: UTC
    """
    if l is None or b is None:
        return None, None

    c = SkyCoord(l=float(l)*u.degree, b=float(b)*u.degree, frame='galactic')
    return c.icrs.ra.deg, c.icrs.dec.deg 

def define_time_intervals(tmin, tmax, binsz=None, nbins=None):
    """
    Define time intervals by either a bin size or the number of bins
    
    Args:
        tmin (float): minimum time
        tmax (float): maximum time
        binsz (astropy.Quantity, optional): bin size with astropy.units
        nbins (int, optional): the number of bins
    
    Return:
        list: time interval (astropy.Time)
    """
    if binsz is not None:
        _, tmin = MJD2UTC(tmin, return_astropy=True)
        _, tmax = MJD2UTC(tmax, return_astropy=True)
        nbins = int((tmax-tmin)*u.second/binsz.to(u.second))
        nbins +=1
        times = tmin + np.arange(nbins) * binsz
    elif nbins is not None:
        _, tmin = MJD2UTC(tmin, return_astropy=True)
        _, tmax = MJD2UTC(tmax, return_astropy=True)
        binsz = ((tmax-tmin)/nbins).to(u.second)
        nbins +=1
        times = tmin + np.arange(nbins) * binsz

    if len(times) == 1:
        time_intervals = [Time([tmin, tmax])]
    else:
        time_intervals = [Time([tstart, tstop]) for tstart, tstop in zip(times[:-1], times[1:])]

    return time_intervals

def generatePSF(config, **kwargs):

    emin = kwargs.pop("emin", config['selection']['emin'])
    emax = kwargs.pop("emax", config['selection']['emax'])
    binsperdec = config['binning']['binsperdec']
    enumbins = kwargs.pop("enumbins", int((np.log10(emax)-np.log10(emin))*binsperdec))
    ntheta = int(30/config['binning']['binsz'])
    thetamax = config['binning']['binsz']*ntheta
    
    from GtApp import GtApp
    gtpsf = GtApp('gtpsf')
    workdir = config['fileio']['workdir']
    gtpsf["expcube"] = '{}/ltcube_00.fits'.format(workdir)
    gtpsf["outfile"] = kwargs.pop("outfile", '{}/gtpsf_00.fits'.format(workdir))
    gtpsf["irfs"] = config['gtlike']['irfs']
    gtpsf['evtype'] = config['selection']['evtype']
    gtpsf['ra'] = config['selection']['ra']
    gtpsf['dec'] = config['selection']['dec']
    gtpsf['emin'] = emin
    gtpsf['emax'] = emax
    gtpsf['thetamax'] = thetamax
    gtpsf['ntheta'] = ntheta
    gtpsf['nenergies'] = enumbins
    gtpsf['chatter'] = 0
    gtpsf.run()

def generateRSP(config):
    from gt_apps import rspgen
    workdir = config['fileio']['workdir']
    emin = config['selection']['emin']
    emax = config['selection']['emax']
    binsperdec = config['binning']['binsperdec']
    enumbins = int((np.log10(emax)-np.log10(emin))*binsperdec)
    
    rspgen['respalg'] = 'PS'
    rspgen['specfile'] = '{}/gtpha_00.pha'.format(workdir)
    rspgen['scfile'] = config['data']['scfile']
    rspgen['outfile'] = '{}/gtrsp_00.rsp'.format(workdir)
    rspgen['thetacut'] = config['selection']['zmax']
    rspgen['irfs'] = config['gtlike']['irfs']
    rspgen['emin'] = config['selection']['emin']
    rspgen['emax'] = config['selection']['emax']
    rspgen['ebinalg'] = "LOG"
    rspgen['enumbins'] = enumbins
    rspgen['chatter'] = 0
    rspgen.run() 

def generatePHA(config):
    from gt_apps import evtbin
    workdir = config['fileio']['workdir']
    emin = config['selection']['emin']
    emax = config['selection']['emax']
    binsperdec = config['binning']['binsperdec']
    enumbins = int((np.log10(emax)-np.log10(emin))*binsperdec)
    
    evtbin['evfile'] = '{}/ft1_00.fits'.format(workdir)
    evtbin['scfile'] = config['data']['scfile']
    evtbin['outfile'] = '{}/gtpha_00.pha'.format(workdir)
    evtbin['algorithm'] = 'PHA1'
    evtbin['ebinalg'] = 'LOG'
    evtbin['emin'] = config['selection']['emin']
    evtbin['emax'] = config['selection']['emax']
    evtbin['enumbins'] = enumbins
    evtbin['coordsys'] = 'CEL'
    evtbin['xref'] = config['selection']['ra']
    evtbin['yref'] = config['selection']['dec']
    evtbin['chatter'] = 0
    evtbin.run()


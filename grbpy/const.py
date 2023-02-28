from pathlib import Path
import astropy.units as u

SCRIPT_DIR = str(Path(__file__).parent.absolute())

keV2Erg = u.keV.to(u.erg)
MeV2Erg = u.MeV.to(u.erg)
GeV2Erg = u.GeV.to(u.erg)
TeV2Erg = u.TeV.to(u.erg)

sec2day = 1./(60*60*24)
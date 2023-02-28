import numpy as np
from astropy.io import fits
from astropy.table import Table

def read_gbm_data(file_name, energy_band=[50, 300], t_shift=0):
    file_name = np.atleast_1d(file_name)

    raw_events = []
    for i, file in enumerate(file_name):
        with fits.open(file) as sourceFile:
            data = Table(sourceFile[2].data)
            pha = Table(sourceFile[1].data)
            e_filter = [128-sum(pha["E_MIN"] > energy) for energy in energy_band]
        
            data['TIME'] = data['TIME']+t_shift

            data = data[data['PHA'] >= e_filter[0]]
            data = data[data['PHA'] <= e_filter[1]+1]
            raw_events += list(data["TIME"])
            
    return np.asarray(raw_events)
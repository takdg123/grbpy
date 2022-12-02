import numpy as np


import matplotlib.pyplot as plt

from . import utils

from .config import InitConfig

from scipy.stats import chi2
from scipy.stats import norm

import astropy.units as u

from astropy.io import fits



def fermi_plotter(name, fermi, subplot = None, **kwargs):
    """
    This returns various plots generated from fermipy.ROIPlotter and fermipy.SEDPlotter

    Args:
        name (str): a plot to show
            Options: ["sqrt_ts", "npred", "ts_hist",
                      "data", "model", "sigma",
                      "excess", "resid", "sed"]
            Default: config.yaml
        fermi (class): vtspy.analysis.FermiAnalysis
        subplot: location of a plot
            Default: None
        **kwargs

    Return:
        AxesSubplot
    """
    from fermipy.plotting import ROIPlotter, SEDPlotter

    output = fermi.output
    roi = fermi.gta.roi
    config = fermi.gta.config

    if subplot is None:
        fig = plt.figure(figsize=(14,6))
        subplot = "111"
        ax = plt.gca()
    else:
        ax = None

    kwargs.setdefault('cmap', config['plotting']['cmap'])

    if name == "sqrt_ts":
        sigma_levels = [3, 5, 7] + list(np.logspace(1, 3, 17))
        plotter = ROIPlotter(output['ts']['sqrt_ts'], roi=roi, **kwargs)
        plotter.plot(
            vmin=0, vmax=5, levels=sigma_levels,
            cb_label='Sqrt(TS) [$\sigma$]',
            interpolation='bicubic', subplot=subplot)
        ax = plt.gca()
        ax.set_title('Sqrt(TS)')

    if name == "npred":
        plotter = ROIPlotter(output['ts']['npred'], roi=roi, **kwargs)
        plotter.plot(
            vmin=0, cb_label='NPred [Counts]', interpolation='bicubic', subplot=subplot)
        ax = plt.gca()
        ax.set_title('NPred')

    if name == "ts_hist":
        ax = plot_ts_hist(output, subplot)
        ax.set_title('TS histogram')

    if name == "data":
        plotter = ROIPlotter(output['resid']['data'],roi=roi)
        plotter.plot(vmin=50,vmax=400,subplot=subplot,cmap='magma')
        ax = plt.gca()
        ax.set_title('Data')

    if name == "model":
        plotter = ROIPlotter(output['resid']['model'],roi=roi)
        plotter.plot(vmin=50,vmax=400,subplot=subplot,cmap='magma')
        ax = plt.gca()
        ax.set_title('Model')

    if name == "sigma":
        plotter = ROIPlotter(output['resid']['sigma'],roi=roi)
        plotter.plot(vmin=-5,vmax=5,levels=[-5,-3,3,5],subplot=subplot,cmap='RdBu_r')
        ax = plt.gca()
        ax.set_title('Significance')

    if name == "excess":
        plotter = ROIPlotter(output['resid']['excess'],roi=roi)
        plotter.plot(vmin=-100,vmax=100,subplot=subplot,cmap='RdBu_r')
        ax = plt.gca()
        ax.set_title('Excess')

    if name == "resid":
        ax = plot_sigma_hist(output, subplot)
        ax.set_title('Residual histogram')

    if name == "sed":
        kwargs.pop('cmap')
        plot_sed(output, **kwargs)
        ax = plt.gca()

    return ax

def plot_ts_hist(output, subplot=None):
    if subplot is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = plt.gcf()
        ax = fig.add_subplot(subplot)

    bins = np.linspace(0, 25, 101)

    data = np.nan_to_num(output['ts']['ts'].data.T)
    data[data > 25.0] = 25.0
    data[data < 0.0] = 0.0
    n, bins, patches = ax.hist(data.flatten(), bins, density=True,
                               histtype='stepfilled',
                               facecolor='green', alpha=0.75)

    ax.plot(bins, 0.5 * chi2.pdf(bins, 1.0), color='k',
            label=r"$\chi^2_{1} / 2$")
    ax.set_yscale('log')
    ax.set_ylim(1E-4)
    ax.legend(loc='upper right', frameon=False)

    # labels and such
    ax.set_xlabel('TS')
    ax.set_ylabel('Probability')

    return ax

def plot_sigma_hist(output, subplot=None):
    if subplot is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = plt.gcf()
        ax = fig.add_subplot(subplot)

    nBins = np.linspace(-6, 6, 121)
    data = np.nan_to_num(output['resid']['sigma'].data)

    # find best fit parameters
    mu, sigma = norm.fit(data.flatten())

    # make and draw the histogram
    data[data > 6.0] = 6.0
    data[data < -6.0] = -6.0

    n, bins, patches = ax.hist(data.flatten(), nBins, density=True,
                               histtype='stepfilled',
                               facecolor='green', alpha=0.75)
    # make and draw best fit line
    y = norm.pdf(bins, mu, sigma)
    ax.plot(bins, y, 'r--', linewidth=2, label="Best-fit")
    y = norm.pdf(bins, 0.0, 1.0)
    ax.plot(bins, y, 'k', linewidth=1, label=r"$\mu$ = 0, $\sigma$ = 1")

    # labels and such
    ax.set_xlabel(r'Significance ($\sigma$)')
    ax.set_ylabel('Probability')
    paramtext = 'Gaussian fit:\n'
    paramtext += '$\\mu=%.2f$\n' % mu
    paramtext += '$\\sigma=%.2f$' % sigma
    ax.text(0.05, 0.95, paramtext, verticalalignment='top',
            horizontalalignment='left', transform=ax.transAxes)

    ax.legend()

    return ax


def plot_ROI(veritas=None, fermi=None):
    plt.figure(figsize=(7, 7))

    if veritas is not None:
        ax = veritas._exclusion_mask.plot()
        veritas._on_region.to_pixel(ax.wcs).plot(ax=ax, edgecolor="red")
        plot_spectrum_datasets_off_regions(ax=ax, datasets=veritas.datasets)

    if fermi is not None:
        if veritas is not None:
            geom = Map.create(npix=(150, 150), binsz=0.05, skydir=fermi.target.skydir, proj="CAR", frame="icrs")
            _, ax, _ = geom.plot()
            ax.add_patch(Patches.Rectangle((0, 0), 150, 150,  color="w"))
        #fermi._src_in_roi(ax)

    #plt.show(block=False)

def plot_sed(output, show_model=True, show_band=True, show_flux_points=True, erg=False, units="MeV", color="k", **kwargs):

    sed = output["sed"]
    fermi_model = output["sed"]['model_flux']
    m_engs = 10**fermi_model['log_energies']*u.MeV
    dnde = fermi_model['dnde']*(1/u.MeV/u.cm**2/u.second)
    dnde_lo = fermi_model['dnde_lo']*(1/u.MeV/u.cm**2/u.second)
    dnde_hi = fermi_model['dnde_hi']*(1/u.MeV/u.cm**2/u.second)

    if units == "MeV":
        energy_units = u.MeV
    elif units == "GeV":
        energy_units = u.GeV
    elif units == "TeV":
        energy_units = u.TeV

    if erg:
        flux_units = u.erg/u.cm**2/u.second
        flux_label = "erg/cm$^2$/s"
    else:
        flux_units = energy_units/u.cm**2/u.second
        flux_label = f"{units}/cm$^2$/s"

    ul_ts_threshold = kwargs.pop('ul_ts_threshold', 9)
    m = sed['ts'] < ul_ts_threshold
    x = (sed['e_ctr']*u.MeV).to(energy_units)
    y = (sed['e2dnde']*u.MeV/u.cm**2/u.second).to(flux_units)

    yerr = (sed['e2dnde_err']*u.MeV/u.cm**2/u.second).to(flux_units)
    yerr_lo = (sed['e2dnde_err_lo']*u.MeV/u.cm**2/u.second).to(flux_units)
    yerr_hi = (sed['e2dnde_err_hi']*u.MeV/u.cm**2/u.second).to(flux_units)
    yul = (sed['e2dnde_ul95']*u.MeV/u.cm**2/u.second).to(flux_units)
    delo = ((sed['e_ctr'] - sed['e_min'])*u.MeV).to(energy_units)
    dehi = ((sed['e_max'] - sed['e_ctr'])*u.MeV).to(energy_units)
    xerr0 = np.vstack((delo[m], dehi[m]))
    xerr1 = np.vstack((delo[~m], dehi[~m]))

    if show_flux_points:
        plt.errorbar(x[~m], y[~m], xerr=xerr1, label="Fermi-LAT",
                     yerr=(yerr_lo[~m], yerr_hi[~m]), ls="", color=color)
        plt.errorbar(x[m], yul[m], xerr=xerr0,
                     yerr=yul[m] * 0.2, uplims=True, ls="", color=color)

    if show_model:
        plt.plot(m_engs.to(energy_units), (dnde*m_engs**2).to(flux_units), color="gray", **kwargs)
        if show_band:
            plt.fill_between(m_engs.value, 
                (dnde_lo * m_engs**2).value, 
                (dnde_hi * m_engs**2).value,
            alpha=0.2, color=color)

    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=13)
    plt.grid(which="major", ls="-")
    
    plt.xlabel(f"Energy [{units}]", fontsize=13)
    plt.ylabel(f"Energy flux [{flux_label}]", fontsize=13)

    fig = plt.gcf()
    return fig

def plot_lc(lc, ts_cut = 9, ax = None, **kwargs):
    if ax is None:
        ax = plt.gca()
    consts = lc["ts"] > ts_cut
    prop = ax.errorbar(lc["time"][consts], lc["e2dnde"][consts], 
             xerr = [lc["time"][consts]-lc["t_min"][consts], lc["t_max"][consts]-lc["time"][consts]], 
             yerr=[lc["e2dnde_err_lo"][consts], lc["e2dnde_err_hi"][consts]], ls="", **kwargs)
    ax.errorbar(lc["time"][~consts], lc["e2dnde_ul95"][~consts], 
                 xerr = [lc["time"][~consts]-lc["t_min"][~consts], lc["t_max"][~consts]-lc["time"][~consts]], 
                 yerr = lc["e2dnde_ul95"][~consts]*0.2, c=prop[0].get_color(),
                 uplims=True, ls="")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time since trigger [s]")
    ax.set_ylabel(r"Energy flux [MeV/cm$^{2}$/s]")
    ax.grid(which="major")
    ax.grid(which="minor", ls=":", alpha=0.5)

def plot_cnt_lc(event=None, binsz=10, c=None, t_shift = 0, units="MeV", cnt_only=False):
    
    if event is None:
        return
    elif type(event) == str:
        config = InitConfig.get_config(config_file=event)
        with open(config["data"]["evfile"]) as f:
            event = fits.open(f.readlines()[0][:-1])[1].data
    
    elif type(event) == InitConfig:
        with open(event.config["data"]["evfile"]) as f:
            event = fits.open(f.readlines()[0][:-1])[1].data

    ax = plt.gca()
    if not(cnt_only):
        ax2 = ax.twinx()
        if units == "GeV":
            event_e = event["ENERGY"]/1e3
            ax2.set_ylabel("Energy [GeV]", fontsize=15)
        else:
            event_e = event["ENERGY"]
            ax2.set_ylabel("Energy [MeV]", fontsize=15)

    if c is not None:
        cm = plt.cm.get_cmap('YlOrBr')
        cbaxes = plt.gcf().add_axes([1, 0.1, 0.02, 0.9]) 
    else:
        c = "gray"
        cm = None
        cmap = None
        cbaxes = None
    
    event_t = event["TIME"] - t_shift
    
    if not(cnt_only):
        sc = ax2.scatter(event_t, event_e, c=c, cmap=cm, alpha=0.5, zorder=-1)
        if cbaxes is not None:
            plt.colorbar(sc, cax=cbaxes, label="Probability")
        ax2.set_yscale("log")
        ax2.grid(which="major", ls="-")
        ax2.grid(which="minor", ls="--", alpha=0.3)
    
    ax.hist(event_t, bins = np.arange(min(event_t), max(event_t)+binsz, binsz), zorder=1)
    
    if t_shift == 0:
        ax.set_xlabel("Time since MET_0 [s]", fontsize=15)
    else:
        ax.set_xlabel("Time since trigger [s]", fontsize=15)

    ax.set_ylabel("Counts", fontsize=15)
    
    ax.grid(which="major", ls="-", axis="x")
    plt.tight_layout()
    return ax, event


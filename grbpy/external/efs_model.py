import numpy as np
import matplotlib.pyplot as plt

lab = [
    r'{:<21s}  $ISM/Wind$'.format(r'$\nu_c < \nu < \nu_m$'),
    r'{:<21s}   $ISM/Wind$'.format(r'$\nu > \nu_m, \nu_c$'),
    r'{:<21s}  $ISM$'.format(r'$\nu_m < \nu < \nu_c$'),
    r'{:<21s}  $Wind$'.format(r'$\nu_m < \nu < \nu_c$'),
    r'{:<21s}   $ISM$'.format(r'$\nu > \nu_m, \nu_c$'),
    "",
    r'{:<21s}   $Wind$'.format(r'$\nu > \nu_m, \nu_c$'),
    ""
    ]

def cr_vals(p, regime):
    if regime == 1:
        return p/2., (3*p-2)/4.
    elif regime == 2:
        return (p-1)/2., 3.*(p-1)/4.

def cr_plot(alpha=None, beta=None, alpha_err=None, beta_err=None, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5,4))

    ax.errorbar(beta, alpha, xerr=beta_err, yerr=alpha_err, ls="", marker="x")
    
    ax.set_xlabel(r"$\beta$", fontsize=12)
    ax.set_ylabel(r"$\alpha$", fontsize=12) 

    ax.set_xlim(0,4)
    ax.set_ylim(0,4)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3, 4])

    ax.plot(*cr_vals(np.linspace(2, 10, 10), 1), color='r', ls='-', label=lab[1])
    ax.plot(np.linspace(1, 2, 10)/2, (3*np.linspace(1, 2, 10)+10)/16., color='r', ls='--', label=lab[4])
    ax.plot(np.linspace(1, 2, 10)/2, (np.linspace(1, 2, 10)+6)/8., color='r', ls=':', label=lab[6])

    ax.plot(*cr_vals(np.linspace(2, 10, 10), 2), color='g', ls='--', label=lab[2])
    ax.plot((np.linspace(1, 2, 10)-1)/2., 3.*(np.linspace(1, 2, 10)+2)/16., color='g', ls='--')

    ax.plot((np.linspace(2, 10, 10)-1)/2., (3.*np.linspace(2, 10, 10)-1)/4., color='g', ls=':', label=lab[3])
    ax.plot((np.linspace(1, 2, 10)-1)/2., (np.linspace(1, 2, 10)+8)/8., color='g', ls=':')
    
    ax.plot(0.5, 0.25, marker='o', ls="", markersize=5, color='orange', markeredgecolor='orange', label=lab[0])
    
    ax.plot(np.linspace(0, 4, 10), np.linspace(0, 4, 10)+2, color='k', ls='-', label="HLE")

    ax.grid(which="major", ls="-")
    ax.legend(bbox_to_anchor=(1.01, -0.01, 0, 1), bbox_transform=ax.transAxes)
    return ax
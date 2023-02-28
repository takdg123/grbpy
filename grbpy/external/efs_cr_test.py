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

def k_from_slow_cooling(beta, beta_err, alpha, alpha_err):
    k = 4*(2*alpha-3*beta)/(2*alpha-3*beta+1)
    k_a_e = 8/(2*alpha-3*beta+1)**2.
    k_b_e = -12/(2*alpha-3*beta+1)**2.
    k_e = np.sqrt(k_a_e**2.*alpha_err**2+k_b_e**2.*beta_err**2)
    return k, k_e

def k_from_nu_c(t1, t2, nu1, nu2):
    dt = np.log10(t2/t1)
    dnu = np.log10(nu2/nu1)
    x = dnu/dt
    k = lambda x: 4*(2*x+1)/(2*x+3)
    return k(x)

def p_from_dF(t1, t2, F1, F2, k=2):
    dF = np.log10(F2/F1)
    dT = np.log10(t2/t1)
    x = dF/dT
    p = (4*(4-k)*x + 5*k - 12)/(3*k-12)
    return p

def k_from_dF(t1, t2, F1, F2, p=2):
    dF = np.log10(F2/F1)
    dT = np.log10(t2/t1)
    x = dF/dT
    k = (16*x+12*p-12)/(3*p+4*x-5)
    return k


def num2condition(num):
    if num == 1:
        return "fast", None, False
    elif num == 2:
        return "slow", "ism", False
    elif num == 3:
        return "slow", "wind", False
    elif num == 4:
        return "fast", "ism", True
    elif num == 5:
        return "fast", "wind", True
    elif num == 6:
        return "slow", "ism", True
    elif num == 7:
        return "slow", "wind", True

def cr_indices(p, cooling, medium=None, low_p=False):
    if type(cooling) == int:
        cooling, medium, low_p = num2condition(cooling)

    if low_p:
        if cooling == "fast" and medium.lower() == "ism":
            return p/2., (3*p+10)/16.
        elif cooling == "fast" and medium.lower() == "wind":
            return p/2., (p+6)/8.
        elif cooling == "slow" and medium.lower() == "ism":
            return (p-1)/2., 3.*(p+2)/16.
        elif cooling == "slow" and medium.lower() == "wind":
            return (p-1)/2., (p+8)/8.

    else:
        if cooling == "fast":
            return p/2., (3*p-2)/4.
        elif cooling == "slow" and medium.lower() == "ism":
            return (p-1)/2., 3.*(p-1)/4.
        elif cooling == "slow" and medium.lower() == "wind":
            return (p-1)/2., (3*p-1)/4.

def cr_alpha2p(alpha=None, alpha_err=None, cooling=None, medium=None):
    if type(cooling) == int:
        cooling, medium, low_p = num2condition(cooling)
    
    if cooling == "fast":
        p = 4./3*alpha+2/3.
    elif cooling == "slow" and medium.lower() == "ism":
        p = 4./3*alpha+1
    elif cooling == "slow" and medium.lower() == "wind":
        p = 4./3*alpha+1/3.
    
    p_err = 4./3*alpha_err

    if p<2:
        if cooling == "fast" and medium.lower() == "ism":
            p = (16*alpha-10)/3.
            p_err = 16./3*alpha_err
        elif cooling == "fast" and medium.lower() == "wind":
            p = 8.*alpha-6
            p_err = 8*alpha_err
        elif cooling == "slow" and medium.lower() == "ism":
            p = 16./3*alpha-2
            p_err = 16./3*alpha_err
        elif cooling == "slow" and medium.lower() == "wind":
            p = 8.*alpha-8
            p_err = p_err = 8*alpha_err

    return p, p_err
        
def cr_beta2p(beta=None, beta_err=None, cooling=None, medium=None):
    if type(cooling) == int:
        cooling, medium, low_p = num2condition(cooling)
        
    if cooling == "fast":
        p = 2*beta
    elif cooling == "slow":
        p = 2*beta+1
    p_err = 2*beta_err
    return p, p_err

def cr_p_vals(beta=None, alpha=None, beta_err=None, alpha_err=None, cooling=None, medium=None):
    p_b, p_b_err = cr_beta2p(beta=beta, beta_err=beta_err, cooling=cooling, medium=medium)
    p_a, p_a_err = cr_alpha2p(alpha = alpha, alpha_err=alpha_err, cooling=cooling, medium=medium)
    return p_b, p_b_err, p_a, p_a_err

def cr_plot(beta=None, alpha=None, beta_err=None, alpha_err=None, ax=None, plot_equal_p = True, p_values=None, **kwargs):
    
    show_cr_legend = kwargs.pop("show_cr_legend", True)

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5,4))

        ax.plot(*cr_indices(np.linspace(2, 10, 10), "fast"), color='r', ls='-', lw=0.7, label=lab[1])
        ax.plot(*cr_indices(np.linspace(1, 2, 10), "fast", "ism", low_p=True), color='r', ls='--', lw=0.7, label=lab[4])
        ax.plot(*cr_indices(np.linspace(1, 2, 10), "fast", "wind", low_p=True), color='r', ls=':', lw=0.7, label=lab[6])

        ax.plot(*cr_indices(np.linspace(2, 10, 10), "slow", "ism"), color='g', ls='--', lw=0.7, label=lab[2])
        ax.plot(*cr_indices(np.linspace(1, 2, 10), "slow", "ism", low_p=True), color='g', lw=0.7, ls='--')

        ax.plot(*cr_indices(np.linspace(2, 10, 10), "slow", "wind"), color='g', ls=':', lw=0.7, label=lab[3])
        ax.plot(*cr_indices(np.linspace(1, 2, 10), "slow", "wind", low_p=True), color='g', lw=0.7, ls=':')
        
        ax.plot(0.5, 0.25, marker='o', ls="", markersize=5, color='pink', markeredgecolor='pink', lw=0.7, label=lab[0])
        
#        ax.plot(np.linspace(0, 4, 10), np.linspace(0, 4, 10)+2, color='k', ls='-', lw=0.7, label="HLE")

        if plot_equal_p:
            if p_values is None:
                p_values = [2.1, 2.5, 2.9, 3.3]
            else:
                p_values = np.atlease1d(p_values)
            for p in p_values:
                vals = np.asarray([cr_indices(p, 1), cr_indices(p, 2), cr_indices(p, 3)])
                plt.plot(vals[:,0], vals[:,1], color="gray", lw=0.5, ls="-.")

    if alpha_err is None and beta_err is None:
        prop = ax.errorbar(beta, alpha, ls="", **kwargs)
    else:
        prop = ax.errorbar(beta, alpha, xerr=beta_err, yerr=alpha_err, ls="", **kwargs)
    
    ax.set_xlabel(r"$\beta$", fontsize=12)
    ax.set_ylabel(r"$\alpha$", fontsize=12) 

    ax.set_xlim(0,4)
    ax.set_ylim(0,4)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3, 4])

    ax.grid(which="major", ls="-")

    if show_cr_legend:
        plt.legend(bbox_to_anchor=(1.01, -0.01, 0, 1), bbox_transform=ax.transAxes)
    
    return ax, prop
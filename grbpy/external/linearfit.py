from iminuit import Minuit
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models

linear = lambda x, m, k: m*x+k

class LinearFit:
    
    def __init__(self, x, y, x_err = [], y_err = [], logx = False, logy = False, model="linear", pinit = None, verbose=False, ext = False):

        self._ext = ext
        self._x = np.asarray(x)

        if len(x) != len(x_err):
            self._x_err = np.zeros(len(x))
        else:
            self._x_err = np.asarray(x_err)

        self._y = np.asarray(y)

        if len(y) != len(y_err):
            self._y_err = np.ones(len(y))
        else:
            self._y_err = np.asarray(y_err)
        
        self._pinit = {"m": -2.6, 
            "m2": -1.3, 
            "bk": 3.3,
            "k": -1,
            "s": -2,
            "sig": 1}

        self._logx=logx
        self._logy=logy
        self._model = model
        self._verbose = verbose

        if pinit is not None:
            for key in pinit.keys():
                self._pinit[key] = pinit[key]
        
        if self.logx and self._model !="sbkn": 
            self._x, self._x_err = self._covLog(self._x, self._x_err)
        elif self._model == "sbkn":
            self._logx=False
            self._logy=False

        if self.logy and self._model !="sbkn": 
            self._y, self._y_err = self._covLog(self._y, self._y_err)

        
        
    @property
    def p(self):
        return self._p

    @property
    def cov(self):
        return self._cov

    @property
    def stat(self):
        return self._stat
    
    @property
    def perr(self):
        return self._perr

    @property
    def logx(self):
        return self._logx

    @property
    def logy(self):
        return self._logy


    def model(self, x):

        if self._model == "linear" or self._model == "pl" :
            func = self._linear
        elif self._model == "bkn":
            func = self._bkn
        elif self._model == "sbkn":
            func = self._sbkn
        
        if hasattr(self, "p"):
            if self.logx and self.logy:
                return 10**(func(np.log10(x), *self.p))
            elif self.logx:
                return func(np.log10(x), *self.p)
            elif self.logy:
                return 10**func(x, *self.p)
            else:
                return func(x, *self.p)
        else:
            return None

    def _covLog(self, t, dt):
        dt = np.log10(t+abs(dt))-np.log10(t)
        t = np.log10(t)
        return t, dt
    
    def _invLog(self, t, dt):
        t = 10**t
        dt = t*(10**dt-1)
        return t, dt
        
    def _linear_ext_likelihood(self, m, k, sig_ext):
        likelihood = 1/2.*sum(np.log(m**2.*self._fx_err**2.+self._fy_err**2.+sig_ext**2.))+1/2.*sum((self._fy-self._linear(self._fx, m, k))**2./(sig_ext**2.+self._fy_err**2.+m**2.*self._fx_err**2.))
        return likelihood

    def _linear_likelihood(self, m, k):
        likelihood = sum((self._fy-self._linear(self._fx, m, k))**2./(self._fy_err**2.+m**2.*self._fx_err**2.))
        return likelihood

    def _bkn_likelihood(self, m, m2, k, bk):
        low = self._fx <= bk
        my = self._bkn(self._fx, m, m2, k, bk)
        likelihood = sum((self._fy[low]-my[low])**2./(self._fy_err[low]**2.+m**2.*self._fx_err[low]**2.))
        likelihood = likelihood+sum((self._fy[~low]-my[~low])**2./(self._fy_err[~low]**2.+m2**2.*self._fx_err[~low]**2.))

        return likelihood

    def _sbkn_likelihood(self, m, m2, k, bk, s):
        tol=1e-4

        if np.isnan(k) or s < -3:
            derv = np.asarray([0]*len(self._fx))
        else:
            x_h = self._fx*(1+tol)
            x_l = self._fx*(1-tol)
            y_h = self._sbkn(x_h, m, m2, k, bk, s)
            y_l = self._sbkn(x_l, m, m2, k, bk, s)
            derv = abs(y_h-y_l)/(2*tol)

        likelihood = sum((self._fy-self._sbkn(self._fx, m, m2, k, bk, s))**2./(self._fy_err**2.+derv**2*self._fx_err**2.))
        
        return likelihood

    @staticmethod
    def _linear(x, m, k):
        return m*x+k
    
    @staticmethod
    def _bkn(x, m, m2, k, bk):
        temp = np.zeros(len(x))
        for i, x_c in enumerate(x):
            if x_c <= bk:
                temp[i] = m*x_c+k
            else:
                temp[i] = m2*x_c+k+(m-m2)*bk
        return temp

    @staticmethod
    def _sbkn(x, m, m2, k, bk, s):
        if np.isnan(k):
            return np.asarray([0]*len(x))
        else:
            
            bk = 10**bk
            k = 10**k
            

            if k == 0 or s < -3:
                return np.asarray([0]*len(x))
            else:
                s = 10**s

                f = models.SmoothlyBrokenPowerLaw1D(amplitude=k, x_break=bk,
                                         alpha_1=-m, alpha_2=-m2, delta=s)
                return f(x)

    def fit(self, start = -1, end = -1, model = None, pinit=None):

        if model is not None:
            if self._model != model:
                self._model = model
                
                if model == "sbkn":
                    if self._logx:
                        self._x, self._x_err = self._invLog(self._x, self._x_err)
                    if self._logy:
                        self._y, self._y_err = self._invLog(self._y, self._y_err)    
                    self._logx = False
                    self._logy = False

        if pinit is not None:
            for key in pinit.keys():
                self._pinit[key] = pinit[key]
        
        if start == -1 and end == -1:
            self._fx, self._fy = self._x, self._y
            self._fx_err, self._fy_err = self._x_err, self._y_err
        elif start == -1:
            self._fx, self._fy = self._x[:-end], self._y[:-end]
            self._fx_err, self._fy_err = self._x_err[:-end], self._y_err[:-end]
        elif end == -1:
            self._fx, self._fy = self._x[start:], self._y[start:]
            self._fx_err, self._fy_err = self._x_err[start:], self._y_err[start:]
        else:
            self._fx, self._fy = self._x[start:-end], self._y[start:-end]
            self._fx_err, self._fy_err = self._x_err[start:-end], self._y_err[start:-end]

        if self._ext:
            minuit=Minuit(self._linear_ext_likelihood, 
                  m = self._pinit["m"], 
                  k = self._pinit["k"], 
                  sig_ext = self._pinit["sig"], 
                  )
        elif self._model == "bkn":
            minuit=Minuit(self._bkn_likelihood, 
                  m = self._pinit["m"],
                  m2 = self._pinit["m2"],
                  k = self._pinit["k"],
                  bk = self._pinit["bk"],
                  )
            minuit.limits["bk"] = (min(self._fx)+0.1, max(self._fx)-0.1)
            minuit.values["bk"] = self._pinit["bk"]

        elif self._model == "sbkn":
            minuit=Minuit(self._sbkn_likelihood, 
                  m = self._pinit["m"], 
                  m2 = self._pinit["m2"], 
                  k = self._pinit["k"], 
                  bk = self._pinit['bk'], 
                  s = self._pinit["s"], 
                  )
            minuit.limits["bk"] = (min(np.log10(self._fx))+0.1, max(np.log10(self._fx))-0.1)
            minuit.values["bk"] = self._pinit["bk"]
            minuit.limits["m"] = (-6, -1.4)
            minuit.limits["m2"] = (-5, 0)
            minuit.limits["s"] = (-3, 1)
            

        elif self._model == "linear" or self._model == "pl" :
            minuit=Minuit(self._linear_likelihood, 
                  m = self._pinit["m"],
                  k = self._pinit["k"]
                  )

        minuit.errordef=1
        fit_result = minuit.migrad()
        minuit.hesse()
        
        chisq = fit_result.fval
        dof = len(self._fx) - len(fit_result.parameters)

        self._p = fit_result.values
        self._cov = fit_result.covariance
        self._perr = fit_result.errors
        self._stat = [chisq, dof]
        self._minuit = minuit
        self._fit_result = fit_result
        self.valid = fit_result.valid

    def plot(self, plot_model=True):
        if self.logx:
            x, xerr = self._invLog(self._x, self._x_err)
            fx, fx_err = self._invLog(self._fx, self._fx_err)
            plt.xscale("log")
        else:
            x, xerr = self._x, self._x_err
            fx, fx_err = self._fx, self._fx_err

        if self.logy:
            y, yerr = self._invLog(self._y, self._y_err)
            fy, fy_err = self._invLog(self._fy, self._fy_err)
            plt.yscale("log")
        else:
            y, yerr = self._y, self._y_err
            fy, fy_err = self._fy, self._fy_err


        plt.errorbar(x, y, xerr=xerr, yerr=yerr, ls='')
        plt.errorbar(fx, fy, xerr=fx_err, yerr=fy_err, ls='')
        plt.title(self._p)

        if plot_model:
            plt.plot(x, self.model(x))
        
        plt.xlabel("time")
        plt.ylabel("flux")
        
        if self._model == "sbkn":
            plt.xscale("log")
            plt.yscale("log")

        

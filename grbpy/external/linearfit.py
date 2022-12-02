from iminuit import Minuit
import numpy as np
import matplotlib.pyplot as plt

linear = lambda x, m, k: m*x+k

class LinearFit:
    
    def __init__(self, x, y, x_err = [], y_err = [], logx = False, logy = False, model="linear", pinit = [-2.6, 6.5, 2000, -1.5, 100, 0.8], verbose=False, ext = True):
        self._ext = ext
        self._x = np.asarray(x)
        if len(x) != len(x_err):
            self._x_err = np.ones(len(x))
            self._ext = False
        else:
            self._x_err = np.asarray(x_err)
        self._y = np.asarray(y)
        if len(y) != len(y_err):
            self._y_err = np.ones(len(y))
            self._ext = False
        else:
            self._y_err = np.asarray(y_err)
        self._pinit = pinit
        
        if logx: 
            self._x, self._x_err = self._covLog(self._x, self._x_err)
            self._pinit[2] = np.log10(self._pinit[2])

        if logy: 
            self._y, self._y_err = self._covLog(self._y, self._y_err)

        self._logx=logx
        self._logy=logy
        self._model = model
        self._verbose = verbose
        
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

        if self._model == "linear":
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
        diff = -k*(((self._fx/bk)**(s*m)+(self._fx/bk)**(s*m2))**(-(s+1.)/s)*(m*(self._fx/bk)**(s*m)+m2*(self._fx/bk)**(s*m2)))/self._fx
        likelihood = sum((self._fy-self._sbkn(self._fx, m, m2, k, bk, s))**2./(self._fy_err**2.+diff**2.*self._fx_err**2.))
        return likelihood

    def _linear(self, x, m, k):
        return m*x+k
    
    def _bkn(self, x, m, m2, k, bk):
        temp = np.zeros(len(x))
        for i, x_c in enumerate(x):
            if x_c <= bk:
                temp[i] = m*x_c+k
            else:
                temp[i] = m2*x_c+k+(m-m2)*bk
        return temp

    def _sbkn(self, x, m, m2, k, bk, s):
        return k*((x/bk)**(s*m)+(x/bk)**(s*m2))**(-1./s)

    def fit(self, start = -1, end = -1, model = None):

        if model is not None:
            self._model = model

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
                  m = self._pinit[0], 
                  k = self._pinit[1], 
                  sig_ext = self._pinit[5], 
                  )
        elif self._model == "bkn":
            minuit=Minuit(self._bkn_likelihood, 
                  m = self._pinit[0],
                  m2 = self._pinit[3],
                  k = self._pinit[1],
                  bk = self._pinit[2],
                  )
            minuit.limits["bk"] = (min(self._fx), max(self._fx))
            minuit.values["bk"] = self._pinit[2]

        elif self._model == "sbkn":
            minuit=Minuit(self._sbkn_likelihood, 
                  m = self._pinit[0], 
                  m2 = self._pinit[3], 
                  k = self._pinit[1], 
                  bk = self._pinit[2], 
                  s = self._pinit[4], 
                  )
            minuit.limits["bk"] = (min(self._fx), max(self._fx))
            minuit.values["bk"] = self._pinit[2]

        elif self._model == "linear":
            minuit=Minuit(self._linear_likelihood, 
                  m = self._pinit[0],
                  k = self._pinit[1]
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


    def plot(self, bkn=False, sbkn=False):
        plt.errorbar(self._x, self._y, xerr=self._x_err, yerr=self._y_err, ls='')
        plt.errorbar(self._fx, self._fy, xerr=self._fx_err, yerr=self._fy_err, ls='')
        x_m = np.linspace(min(self._x), max(self._x), 100)
        plt.plot(x_m, self.model(x_m))
        plt.title(self._p)
        
        if sbkn:
            plt.xlabel("time")
            plt.ylabel("flux")
            plt.xscale("log")
            plt.yscale("log")
        else:
            plt.xlabel("log(time)")
            plt.ylabel("log(flux)")
        plt.show(block=False)

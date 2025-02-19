import numpy as np
import sys
sys.path.append('../')
from paths import ML_Path
sys.path.append(ML_Path)
from utilities import Utilities

class Polynomials(Utilities):
    def model_poly(self,x,theta):
        y = np.zeros_like(x)
        for p in range(len(theta)):
            y += theta[p]*x**p
        return y

    def polyfit_custom(self,x,y,deg,sig2=None,start=0):
        """ Polynomial fit of degree deg to data y at locations x.
            Optionally pass squared errors sig2 on y.
            Minimises chi2 = sum_i (y_i - p(x_i))^2 / sig2_i
            with p(x) = sum_alpha a[alpha]*x^alpha
            for alpha=start..deg.
            Returns minimum variance estimator a[alpha] 
            and covariance matrix C[alpha,beta].

            Not very well tested, so use with care. """

        Y = np.zeros(deg+1-start,dtype=float)

        # Matrix
        F = np.zeros((deg+1-start,deg+1-start),dtype=float)

        if sig2 is None:
            sig2 = np.ones(x.size,dtype=float)

        for alpha in range(start,deg+1):
            Y[alpha-start] = np.sum(y*(x**(alpha))/sig2)
            for beta in range(start,deg+1):
                F[alpha-start,beta-start] = np.sum(x**(alpha+beta)/sig2)
                F[beta-start,alpha-start] = F[alpha-start,beta-start]

        Y = Y.T
        Cov,detF = self.svd_inv(F)
        a_minVar = np.dot(Cov,Y)

        return np.squeeze(np.asarray(a_minVar)),np.asarray(Cov)
        

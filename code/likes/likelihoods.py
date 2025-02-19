import numpy as np
import sys
sys.path.append('../')
from paths import ML_Path
sys.path.append(ML_Path)
from utilities import Utilities
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory

#############################################
class Chi2Like(Likelihood,Utilities):
    data = None # expect 1-d array
    cov_mat = None # expect 2-d array
    #########################################
    def initialize(self):
        # various checks on input data and inverse covariance matrix
        if self.data is None:
            raise Exception('data must be specified in Chi2Like.')
        if self.cov_mat is None:
            raise Exception('cov_mat must be specified in Chi2Like.')
        
        if len(self.data.shape) != 1:
            raise Exception('data should be 1-d array in Chi2Like.')
        if len(self.cov_mat.shape) != 2:
            raise Exception('cov_mat should be 2-d array in Chi2Like.')
            
        if self.cov_mat.shape != (self.data.size,self.data.size):
            raise Exception('Incompatible inverse covariance and data in Chi2Like.')
        
        self.invcov_mat,self.det_C = self.svd_inv(self.cov_mat)
        
    #########################################

    #########################################
    def get_requirements(self):
        """ Theory code should return model array. """
        return {'model': None}
    #########################################

    #########################################
    def logp(self,**params_values_dict):
        model = self.provider.get_model()
        residual = self.data - model
        chi2 = np.dot(residual,np.dot(self.invcov_mat,residual))
        return -0.5*chi2
    #########################################

    
#########################################
class PolyTheory(Theory):
    xvals = None # expect 1-d array of same size as data input to likelihood
    #########################################
    def initialize(self):
        if self.xvals is None:
            raise Exception('xvals must be specified in PolyTheory.')
        if len(self.xvals.shape) != 1:
            raise Exception('xvals should be 1-d array in PolyTheory.')
    #########################################

    #########################################
    def calculate(self,state, want_derived=False, **params_values_dict):
        keys = list(params_values_dict.keys())
        params = np.array([params_values_dict[key] for key in keys])
        output = np.sum(np.array([params[p]*self.xvals**p for p in range(len(keys))]),axis=0)
        # parameter dictionary dynamically decides degree of polynomial
        state['model'] = output
    #########################################

    #########################################
    def get_model(self):
        return self.current_state['model']
    #########################################

    #########################################
    def get_allow_agnostic(self):
        return True
    #########################################


#########################################
class GaussTheory(Theory):
    xvals = None # expect 1-d array of same size as data input to likelihood
    #########################################
    def initialize(self):
        if self.xvals is None:
            raise Exception('xvals must be specified in GaussTheory.')
        if len(self.xvals.shape) != 1:
            raise Exception('xvals should be 1-d array in GaussTheory.')
    #########################################

    #########################################
    def pGauss(self,params):
        # expect params = [[amp,mu,sigma]
        ncomp = len(params) // 3
        out = np.zeros_like(self.xvals)
        for n in range(ncomp):
            comp = -0.5*(self.xvals-params[3*n+1])**2/params[3*n+2]**2
            comp = params[3*n]*np.exp(comp)/np.sqrt(2*np.pi)/params[3*n+2]
            out += comp
        return out
    #########################################

    #########################################
    def calculate(self,state, want_derived=False, **params_values_dict):
        keys = list(params_values_dict.keys())
        params = np.array([params_values_dict[key] for key in keys])
        output = self.pGauss(params)
        state['model'] = output
    #########################################

    #########################################
    def get_model(self):
        return self.current_state['model']
    #########################################

    #########################################
    def get_allow_agnostic(self):
        return True
    #########################################
    

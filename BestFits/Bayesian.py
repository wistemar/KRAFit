import itertools

import numpy as np
from threeML import BayesianAnalysis

class Bayesian():
    
    @property
    def percentiles(self):
        return { 'value' : 50 , 'negative_error' : 15.9, 'positive_error' : 84.1 }
    
    
    def bayesian_info(self, sampler, sampler_params):
        
        self.sampler = sampler
        self.sampler_params = sampler_params
        self.method = 'bayesian'
        self.method_short = 'bsn'
        
        return
    
    
    def _perform_fit(self, i):
        
        bsn = BayesianAnalysis(self.model.model, self.model.grb.data[i])
        bsn.set_sampler(self.sampler)
        
        if self.sampler == 'emcee':
            bsn.sampler.setup(n_walkers = self.sampler_params['nw'], n_burn_in = self.sampler_params['nb'], n_iterations = self.sampler_params['ni'])

        elif self.sampler == 'multinest':
            bsn.sampler.setup(n_live_points = self.sampler_params['multi_live'], resume = False)

        elif self.sampler == 'ultranest':
            dlogz = 0.5
            bsn.sampler.setup(min_num_live_points = self.sampler_params['live'], frac_remain = 0.01, use_mlfriends = True, dlogz = dlogz, Lepsilon = 0.001, min_ess = 400)

        else:
            raise ValueError('Chosen sampler is not setup yet.')

        # Perform the fit and get analysis result
        bsn.sample()
        
        return bsn
    
    
    def add_to_df(self, index, ar):
        
        for key, (k, param) in itertools.product(self._results.keys(), enumerate(self.free_parameters)):
            self._results[key].at[index, param] = np.nanpercentile(ar.samples.T[:, k], self.percentiles[key])
            if key != 'value':
                self._results[key].at[index, param] = self._results[key].at[index, param] - self._results['value'].at[index, param]
                    
        super()._add(index, ar)
        
        return
    
    

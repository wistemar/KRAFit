import random

from threeML.io.logging import setup_logger

log = setup_logger(__name__)

from KRAFit.BestFits.JointLikelihood import JL
from KRAFit.BestFits.SimData import SimData

class SimJL(SimData, JL):
    
    def __init__(self, model, num_starts = 1, stat_types = ['Fit'], start_values = {}, **kwargs):
        
        super().jl_info(start_values)
        
        super().__init__(model, num_starts, stat_types, **kwargs)
        
        return
    
    
    def _best_fits_path(self, key, grid_extension) -> str:
        return '{results_path}/best_fit_params/{0}/simulated_data_{method_model}_params_{num_params}_{1}_grid_{2}{3}.csv'.format(self.method, key, grid_extension, self.extension, **self._save_vars)
    
    
    
    def save_vars_extend(self, index, **kwargs):
        
        start_info = 'init_params_' + '_'.join(['{0}_{1}'.format(param, round(self.start_values[param][index[k]], 3)) for k, param in enumerate(self.start_values)])
        
        save_vars_extended = super().save_vars_extend()
        
        save_vars_extended['start_info'] = start_info
        
        save_vars_extended['start_values'] = start_info
        # save_vars_extended['sampler_params'] = ''
            
        return save_vars_extended
        
    
    def _fit_setup(self, index, save_vars_extended, **kwargs):
        
        for k, param in enumerate(self.free_parameters):
            while True:
                try:
                    a = random.uniform(0.9, 1.1)
                    self.model.spectra[param].value = self.start_values[param + 's'][index[k]] * a
                    break
                except:
                    pass
                
            log.info('Start value: {0} = {1}'.format(param, self.model.spectra[param].value))
            
        for param in self.fix_parameters:
                self.model.spectra[param].value = self._results['value'].loc[index, param]
                log.info('Fixed value: {0} = {1}'.format(param, self.model.spectra[param].value))
                
        file_exists, path_fit_file = super()._fit_setup(index, save_vars_extended, **kwargs)
        
        return file_exists, path_fit_file
    
    
    def _perform_fit(self, index):
        i = self._results['value'].index.get_loc(index)
        return self.perform_fit(i)
    
    
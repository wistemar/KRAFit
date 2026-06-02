from threeML.io.logging import setup_logger

log = setup_logger(__name__)

from KRAFit.BestFits.Bayesian import Bayesian
from KRAFit.BestFits.SimData import SimData

class RealBayes(SimData, Bayesian):
    
    def __init__(self, model, num_starts = 1, stat_types = ['Fit'], sampler = None, sampler_params = {}, **kwargs):
        
        super().bayesian_info(sampler, sampler_params)
        
        super().__init__(model, num_starts, stat_types, **kwargs)
        
        self._save_vars['sampler_params'] = '_'.join(['{0}_{1}'.format(param, self.sampler_params[param]) for param in self.sampler_params])
        
        return
    
    
    def _best_fits_path(self, key, save_vars_extended, flags):
        return '{results_path}/best_fit_params/{0}/GRB{number}_{method_model}_params_{num_params}_best_fit_params_{1}_{bin_method}_{bin_var}_time_{start_time}-{stop_time}_{sampler_params}{2}.csv'.format(self.method, key, self.save_extension_setter(flags['save_extension']), **save_vars_extended)
    
    
    def save_vars_extend(self, index, **kwargs):
        
        extension = self._save_vars['sampler_params']
        save_vars_extended = super().save_vars_extend(index, extension)
        
        save_vars_extended['sampler_params'] = extension
        
        return save_vars_extended
  
    
    def _fit_setup(self, index, save_vars_extended, **kwargs):
        
        for param in self.parameters:
            if param in self.free_parameters:
                log.info('Start value: {0} = {1}'.format(param, self.model.spectra[param].value))
            else:
                self.model.spectra[param].value = self._results['value'].loc[index, param]
                log.info('Fixed value: {0} = {1}'.format(param, self.model.spectra[param].value))
                
        file_exists, path_fit_file = super()._fit_setup(index, save_vars_extended, **kwargs)
        
        return file_exists, path_fit_file
    
    
    def _perform_fit(self, index):
        i = self._results['value'].index.get_loc(index)
        return self.perform_fit(i)
    
    
    def _fit_plots(self, index, samples, flags, **kwargs):
        
        super()._fit_plots(index, flags, **kwargs)
            
        flags['converted'] = False
        ar = kwargs.pop('ar', None)
        MAP_points = [ar.get_data_frame()['value'][k] for k, _ in enumerate(self.free_parameters)]
        self.corner(self.free_parameters, index[0], index[1], flags, MAP_points = MAP_points, **kwargs)
        
        if self.sampler == 'emcee':
            self.walker(index[0], index[1], flags)
            
        return 
    
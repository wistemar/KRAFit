import itertools
import os
from abc import abstractmethod

import pandas as pd
from threeML.io.logging import setup_logger

log = setup_logger(__name__)

from KRAFit.BestFits.BestFits import BestFits

class RealData(BestFits):
    
    def __init__(self, model, num_starts, stat_types, **kwargs):
        
        super().__init__(model, num_starts, stat_types, **kwargs)
        
        return
    
    
    def _load(self, flags, **kwargs):
        
        source_start, source_stop = self.full_interval()
        
        for key in self._results:
            
            best_fits_path = self._best_fits_path(key, source_start, source_stop, flags)
            file_exists = os.path.exists(best_fits_path)
            
            if file_exists:
                self._results[key] = pd.read_csv(best_fits_path, index_col = [0, 1, 2])
                    
            else:
                self._create(key, flags, **kwargs)
                
        return
    
    
    def _save_best_fit_params(self, flags, **kwargs) -> None:
        
        source_start, source_stop = self.full_interval()
        
        for key in self._results:
            best_fits_path = self._best_fits_path(key, source_start, source_stop, flags)
            self._results[key].to_csv(best_fits_path, mode = 'w')
            
        return
    
    @abstractmethod
    def _best_fits_path(self, key, save_vars_extended, flags):
        return
    
    
    def full_interval(self):
        
        source_start = self.model.grb.start
        source_stop = self.model.grb.stop
        
        # save_vars_extended = self.save_vars_extend([0, 0], **kwargs)
        # save_vars_extended['start_time'] = round(self.model.grb.detector_data[0].bins[0].start_time, 3)
        # save_vars_extended['stop_time']  = round(self.model.grb.detector_data[0].bins[-1].stop_time, 3)
        
        return source_start, source_stop

    
    def _create(self, key, flags, **kwargs):

        # Needs to be first since it changes num_starts based on the gamma assumption
        if self.model.name == 'KRA' and self.num_params.startswith('5+1'):
            gammas = self.set_gammas(flags, **kwargs)
        else:
            pass
        
        index = pd.MultiIndex.from_product([range(self.model.grb.num_bins), range(self.num_starts), self.stat_types], names = ['Bin', 'Start', 'Type'])
        self._results[key] = pd.DataFrame(columns = self.columns, index = index)
        
        for i, j in itertools.product(range(self.model.grb.num_bins), range(self.num_starts)):
            self._results[key].loc[(i, j, 'Fit'), 'Failed'] = True
            for param in self.fix_parameters:
                if key == 'value':
                    self._results[key].loc[(i, j, 'Fit'), param] = self.model.spectra[param].value
                else:
                    self._results[key].loc[(i, j, 'Fit'), param] = 0.
                   
        # Needs to be last since it uses self._results
        if self.model.name == 'KRA' and self.num_params.startswith('5+1'):
            for i, j, stat_type, param in itertools.product(range(self.model.grb.num_bins), range(self.num_starts), self.stat_types, ['scale', 'G']):
                self._results[key].at[(i, j, stat_type), param] = gammas[key][i, j]   
        else:
            pass
            
        return
    
    
    def fit(self, bins, starts, flags, **kwargs):
        
        bins, starts = self.get_bins_starts(bins, starts)
        
        indices = [index for index in self._results['value'].index if index[0] in bins and index[1] in starts]
        
        loop_description = 'Bins fitted: '
        
        ar = super().fit(indices, loop_description, flags, **kwargs)
        
        return ar
    
    
    def get_bins_starts(self, bins, starts):
        # If bins/starts are not defined, loop over all
        if bins is None or bins == []:
            bins = range(self.model.grb.num_bins)
        if starts == []:
            starts = range(self.num_starts)
        return bins, starts
    
    @abstractmethod
    def save_vars_extend(self, index, start_info):
        
        save_vars_extended = super().save_vars_extend(index)
        
        save_vars_extended['start'] = 'start_{0}'.format(index[1])
        save_vars_extended['start_info'] = '{0}_{1}'.format(save_vars_extended['start'], start_info)
        
        return save_vars_extended
    
    
    def _fit_setup(self, index, save_vars_extended, **kwargs):
        
        
        path_fit_file = '{results_path}/fits_files/{0}/GRB{number}_{method_model}_params_{num_params}_{bin_method}_{bin_var}_time_{start_time}-{stop_time}_{start_info}{1}.fits'.format(self.method, self.save_extension_setter(kwargs.get('save_extension', '')), **save_vars_extended)
        file_exists = os.path.exists(path_fit_file)
     
        log.info('Bin: {0}; Time {start_time}-{stop_time}'.format(index[0], **save_vars_extended))
        log.info('Start: {0}'.format(index[1]))
        
        return file_exists, path_fit_file
    
    @abstractmethod
    def _perform_fit(self):
        pass
    
    
    def _save_model_data(self, index, save_vars_extended) -> None:
        i = index[0]
        return super()._save_model_data(i, save_vars_extended)
    
    
    def _model_data_path(self, detector, save_vars_extended):
        return '{results_path}/model_data/GRB{number}_{method_model}_params_{num_params}_model_data_detector_{0}_{bin_method}_{bin_var}_time_{start_time}-{stop_time}_{start_info}'.format(detector, **save_vars_extended)
    
    
    @abstractmethod
    def _fit_plots(self, index, flags, **kwargs):
        
        if flags.get('view_count_spectra', False) or flags.get('view_walker_path', False):
            log.info('Bin: {0}'.format(index[0]))
            log.info('Start: {0}'.format(index[1]))
            
        super()._fit_plots(index, flags, **kwargs)
    
 
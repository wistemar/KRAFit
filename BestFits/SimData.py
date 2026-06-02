import os
from abc import abstractmethod

import pandas as pd
from threeML.io.logging import setup_logger

log = setup_logger(__name__)

from KRAFit.BestFits.BestFits import BestFits

class SimData(BestFits):
    
    def __init__(self, model, num_starts, stat_types, **kwargs):
        
        super().__init__(model, num_starts, stat_types, **kwargs)
        
        return
    
    
    def create_testing_grid(self, **kwargs):
        
        if not kwargs.get('testing_grid', False):
            raise ValueError('When using fake data, you need to provide a testing grid describing where to generate fake data.')
        testing_grid = kwargs.get('testing_grid')
        
        self.start_values = testing_grid
        
        self.testing_grid = list(testing_grid.values())
        self.n_dims = [len(grid) for grid in self.testing_grid]
        self.num_points = 1
        for grid in self.testing_grid:
            self.num_points *= len(grid)
        
        return
    
    
    def _load(self, flags, **kwargs):
        
        for key in self._results:
            grid_extension = '_'.join(['{0}'.format(val) for val in self.testing_grid])
            best_fits_path = self._best_fits_path(key, grid_extension)
                
            file_exists = os.path.exists(best_fits_path)
            
            if file_exists:
                if flags['is_fake_data']:
                    if self.num_params.startswith('6+0') or self.num_params.startswith('5+1'):
                        self._results[key] = pd.read_csv(best_fits_path, index_col = [0, 1, 2, 3, 4, 5, 6])
                        
                    else:
                        self._results[key] = pd.read_csv(best_fits_path, index_col = [0, 1, 2, 3, 4, 5])
                    
            else:
                self._create(key, flags, **kwargs)
                
        return
    
    
    def _save_best_fit_params(self, flags, **kwargs) -> None:
        
        for key in self._results:
            grid_extension = '_'.join(['{0}'.format(val) for val in self.testing_grid])
            best_fits_path = self._best_fits_path(key, grid_extension)
            self._results[key].to_csv(best_fits_path, mode = 'w')
            
        return
    
    @abstractmethod
    def _best_fits_path(self, key, grid_extension):
        return
    
    
    def _fake_create(self, key: str, **kwargs) -> None:
        
        indices = pd.MultiIndex.from_product([range(n_dim) for n_dim in self.n_dims] + [self.stat_types], names = [param + 's' for param in self.parameters] + ['Type'])
        self._results[key] = pd.DataFrame(columns = self.columns, index = indices)
        
        for index in indices:
            self._results[key].at[(index), 'Failed'] = True
            if index[-1] == 'Initial':
                self._results[key].loc[(index), self.parameters] = [grid[index[k]] for k, grid in enumerate(self.testing_grid)]
        
        return
    
    
    def fit(self, points, flags, **kwargs):
        
        if points == []:
            points = range(self.num_points)
            
        indices = self._results['value'].index[points]
        
        loop_description = 'Points fitted: '
        
        ar = super().fit(indices, loop_description, flags, **kwargs)
        
        return ar
    
    
    def save_vars_extend(self):
        return
    
    
    def _fit_setup(self, index, save_vars_extended, **kwargs):
        
        path_fit_file = '{results_path}/fits_files/{0}/GRB{number}_{method_model}_params_{num_params}_{start_info}{1}.fits'.format(self.method, self.save_extension_setter(kwargs.get('save_extension', '')), **save_vars_extended)
        file_exists = os.path.exists(path_fit_file)
        
        log.info('Point: {0}; Index: {1}'.format(self._results['value'].index.get_loc(index), index))
        init_vals = list(self._results['value'].loc[index, self.free_parameters])
        log.info('Initial: {0}'.format(', '.join(['{0} = {1}'.format(param, init_vals[k]) for k, param in enumerate(self.free_parameters)])))
        
        return file_exists, path_fit_file
    
    
    @abstractmethod
    def _perform_fit(self):
        pass
    
    
    def _save_model_data(self, index, save_vars_extended) -> None:
        i = self._results['value'].index.get_loc(index)
        return super()._save_model_data(i, save_vars_extended)
    
    
    def _model_data_path(self, detector, save_vars_extended):
        return '{results_path}/model_data/GRB{number}_{method_model}_params_{num_params}_new_model_data_detector_{0}_{bin_method}_{bin_var}_time_{start_time}-{stop_time}_{start_info}'.format(detector, **save_vars_extended)
    
    
    
    
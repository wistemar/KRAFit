import copy
import itertools
import os
from abc import ABC, abstractmethod
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpi4py import MPI
from scipy.stats import truncnorm
from threeML import load_analysis_results, display_spectrum_model_counts
from threeML.io.logging import setup_logger
from threeML.utils.progress_bar import tqdm
#from tqdm import tqdm

log = setup_logger(__name__)

from KRAFit.BestFits.Samples import Samples
from KRAFit.Convert import convert
from KRAFit.Misc import misc, constants
from KRAFit.Plots.AIC import AIC
from KRAFit.Plots.Corner import Corner
from KRAFit.Plots.CountSpectra import CountSpectra
from KRAFit.Plots.ParamEvo import ParamEvo
from KRAFit.Plots.ResolvedSpectra import ResolvedSpectra
from KRAFit.Plots.SpectralEvolution import SpectralEvolution


class BestFits(ABC):
    
    def __init__(self, model, num_starts: int, stat_types, **kwargs):
        
        self.model = model # The model with which this fitting is performed
        self.num_starts = num_starts # The different number of start values for each bin
        self.stat_types = stat_types # Different types of results, e.g. for bayesian M.A.P and Mean are not the same
        
        self._results = { # Stores the parameter values from the fits
            'value'         : None,
            'negative_error': None,
            'positive_error': None,
            }
        
        self.parameters = self.model.parameters
        self.free_parameters = [param.split('.')[-1] for param in self.model.model.free_parameters.keys()] # Current free parameters in the model
        self.fix_parameters = [param for param in self.parameters if param not in self.free_parameters] # Current fixed parameters in the model
        
        self.num_params = '{0}+{1}'.format(len(self.free_parameters), len(self.fix_parameters))
        if self.model.name == 'KRA' and self.model.add_powerlaw:
            self.num_params = '{0}+{1}+1'.format(len(self.free_parameters) - 1, len(self.fix_parameters))
        
        self._save_vars = { # Variables used to differentiate the names of different files
            'results_path' : self.model.grb.results_path,
            'number'   : self.model.grb.number,
            'method_model' : '{0}_{1}'.format(self.method_short, self.model.name),
            'num_params'   : '{0}'.format(self.num_params),
            'bin_method'   : self.model.grb.bin_method[0],
            'bin_var'      : self.model.grb.bin_method[1],
        }
        
        if self.method == 'bayesian':
            self._save_vars['sampler_params'] = '_'.join(['{0}_{1}'.format(param, self.sampler_params[param]) for param in self.sampler_params])
            
        self.extension = '_{0}'.format(kwargs.get('extension', '')) if kwargs.get('extension', False) else '' # If there is an extension as to not overwrite something
        
        self.get_df_columns()
        self._load_or_create(**kwargs)
        
        self.num_starts = len(self._results['value'].index.levels[1])
         
        return
    
    
    @property
    def parameters(self):
        return self.model.parameters
    
    
    @property
    def free_parameters(self):
        return self.model.free_parameters
    
    
    @property
    def fix_parameters(self):
        return self.model.fix_parameters
    
    
    @property
    def num_params_sum(self):
        return int(self.num_params[0]) + int(self.num_params[2])
    
    
    @property
    def value(self):
        return self._value()
    
    def _value(self):
        
        rounded_results = copy.deepcopy(self._results['value'])
        for param in rounded_results:
            if param != 'Failed':
                round_number = self.round_number(param)
                rounded_results[param] = rounded_results[param].apply(misc._round, digits = round_number)

        return rounded_results[self.parameters + ['AIC', 'Failed', 'flux', 'pflux']]
    
    
    @property
    def values(self):
        return self._values()
    
    def _values(self):
        
        rounded_results = copy.deepcopy(self._results['value'])
        for param in rounded_results:
            if param != 'Failed':
                round_number = self.round_number(param)
                rounded_results[param] = rounded_results[param].apply(misc._round, digits = round_number)

        return rounded_results
    
    
    @property
    def nerror(self):
        return self._nerror()
    
    def _nerror(self):
        
        rounded_results = copy.deepcopy(self._results['negative_error'])
        for param in rounded_results:
            if param != 'Failed':
                round_number = self.round_number(param)
                rounded_results[param] = rounded_results[param].apply(misc._round, digits = round_number)

        return rounded_results
    
    
    @property
    def perror(self):
        return self._perror()
    
    def _perror(self):
        
        rounded_results = copy.deepcopy(self._results['positive_error'])
        for param in rounded_results:
            if param != 'Failed':
                round_number = self.round_number(param)
                rounded_results[param] = rounded_results[param].apply(misc._round, digits = round_number)

        return rounded_results
    
    
    @property
    def save_vars(self):
        return self._save_vars
    
    
    @property
    def bin_midpoints(self):
        return self.model.grb.detector_data[0].bins.mid_points
    
    
    @property
    def bin_starts(self):
        return np.array(self.model.grb.detector_data[0].bins.starts)
    
    
    @property
    def bin_stops(self):
        return np.array(self.model.grb.detector_data[0].bins.stops)
    
    
    @property
    def bin_counts(self):
        return self.model.grb.detector_data[0].total_counts_per_interval
    
    
    def get_df_columns(self) -> None:
        
        flux_params = constants.constants(['FLUX_PARAMS'])
        
        if self.model.name == 'KRA':
            KRA_params, RMS_params, burst_params = constants.constants(['KRA_PARAMS', 'RMS_PARAMS', 'BURST_PARAMS{0}'.format(self.num_params[2])])
            self.columns = KRA_params + ['AIC', 'Failed'] + RMS_params + burst_params + flux_params
        elif self.model.name == 'Band':
            self.columns = self.parameters + ['AIC', 'Failed'] + flux_params
        elif self.model.name == 'Blackbody':
            self.columns = self.parameters + ['AIC', 'Failed'] + flux_params
        elif self.model.name == 'dsbpl'or self.model.name == 'DoubleSmoothlyBrokenPowerlaw':
            self.columns = self.parameters + ['AIC', 'Failed'] + flux_params
        elif self.model.name == 'NonDissipativePhotosphere':
            self.columns = self.parameters + ['AIC', 'Failed'] + flux_params
        else:
            raise ValueError('Model {name} is not supported yet!'.format(name = self.model.name))
        return
    
    
    def _load_or_create(self, **kwargs):
        
        flags = kwargs.pop('flags', {})
        
        if not kwargs.get('overwrite', False):
            self._load(flags, **kwargs)
            
        else:
            for key in self._results:
                self._create(key, flags, **kwargs)
                
        return
    
    
    # @abstractmethod
    # def _load(self, flags, **kwargs):
        
    #     save_vars_extended = self.full_interval(**kwargs)
        
    #     for key in self._results:
    #         if flags['is_fake_data']:
    #             grid_extension = '_'.join(['{0}'.format(val) for val in self.testing_grid])
    #             best_fits_path = '{results_path}/best_fit_params/{0}/simulated_data_{method_model}_params_{num_params}_{1}_grid_{2}_{sampler_params}{3}.csv'.format(self.method, key, grid_extension, self.extension, **save_vars_extended)
                
    #         elif self.method == 'joint_likelihood':
    #             best_fits_path = '{results_path}/best_fit_params/{0}/GRB{number}_{method_model}_params_{num_params}_best_fit_params_{1}_{bin_method}_{bin_var}_time_{start_time}-{stop_time}{2}.csv'.format(self.method, key, self.extension, **save_vars_extended)
                
    #         elif self.method == 'bayesian':
    #             best_fits_path = '{results_path}/best_fit_params/{0}/GRB{number}_{method_model}_params_{num_params}_best_fit_params_{1}_{bin_method}_{bin_var}_time_{start_time}-{stop_time}_{sampler_params}{2}.csv'.format(self.method, key, self.extension, **save_vars_extended)
                
    #         file_exists = os.path.exists(best_fits_path)
    #         if file_exists:
    #             if flags['is_fake_data']:
    #                 if self.num_params.startswith('6+0') or self.num_params.startswith('5+1'):
    #                     self._results[key] = pd.read_csv(best_fits_path, index_col = [0, 1, 2, 3, 4, 5, 6])
                        
    #                 else:
    #                     self._results[key] = pd.read_csv(best_fits_path, index_col = [0, 1, 2, 3, 4, 5])
                        
    #             else:
    #                 self._results[key] = pd.read_csv(best_fits_path, index_col = [0, 1, 2])
                    
    #         else:
    #             self._create(key, flags, **kwargs)
                
    #     return
    
    # def full_interval(self, **kwargs):
    #     save_vars_extended = self.save_vars_extend(0, 0, **kwargs)
    #     save_vars_extended['start_time'] = round(self.model.grb.detector_data[0].bins[0].start_time, 3)
    #     save_vars_extended['stop_time']  = round(self.model.grb.detector_data[0].bins[-1].stop_time, 3)
    #     return save_vars_extended
    
    
    # @abstractmethod
    # def _create(self, key, flags, **kwargs):
        
    #     if flags['is_fake_data']:
    #         self._fake_create(key, **kwargs)
    #         return
        
    #     # Needs to be first since it changes num_starts based on the gamma assumption
    #     if self.model.name == 'KRA' and self.num_params.startswith('5+1'):
    #         gammas = self.set_gammas(flags, **kwargs)
    #     else:
    #         pass
        
    #     index = pd.MultiIndex.from_product([range(self.model.grb.num_bins), range(self.num_starts), self.stat_types], names = ['Bin', 'Start', 'Type'])
    #     self._results[key] = pd.DataFrame(columns = self.columns, index = index)
        
    #     for i, j in itertools.product(range(self.model.grb.num_bins), range(self.num_starts)):
    #         self._results[key].loc[(i, j, 'Fit'), 'Failed'] = True
    #         for param in self.fix_parameters:
    #             if key == 'value':
    #                 self._results[key].loc[(i, j, 'Fit'), param] = self.model.spectra[param].value
    #             else:
    #                 self._results[key].loc[(i, j, 'Fit'), param] = 0.
                   
    #     # Needs to be last since it uses self._results
    #     if self.model.name == 'KRA' and self.num_params.startswith('5+1'):
    #         for i, j, stat_type, param in itertools.product(range(self.model.grb.num_bins), range(self.num_starts), self.stat_types, ['scale', 'G']):
    #             self._results[key].at[(i, j, stat_type), param] = gammas[key][i, j]
                
    #     else:
    #         pass
            
                                    
    #     return
    
    
    # def _fake_create(self, key: str, **kwargs) -> None:
        
    #     indices = pd.MultiIndex.from_product([range(n_dim) for n_dim in self.n_dims] + [self.stat_types], names = [param + 's' for param in self.parameters] + ['Type'])
    #     self._results[key] = pd.DataFrame(columns = self.columns, index = indices)
        
    #     for index in indices:
    #         self._results[key].at[(index), 'Failed'] = True
    #         if index[-1] == 'Initial':
    #             self._results[key].loc[(index), self.parameters] = [grid[index[k]] for k, grid in enumerate(self.testing_grid)]
        
    #     # index = pd.MultiIndex.from_product([range(self.model.grb.num_bins), range(self.num_starts), self.stat_types], names = ['Bin', 'Start', 'Type'])
        
    #     return
    
    
    def set_gammas(self, flags, **kwargs):
        
        # if gammas is None:
        #     raise ValueError('Gamma needs to be specified when using the 5+1 parameter version.')
        
        result = {}
        gammas = self.load_gammas(flags, flags.get('save_extension', ''))
        
        const_gammas = kwargs.get('const_gammas', False)
        
        if const_gammas and const_gammas != []:
            
            for k, _ in enumerate(gammas):
                gammas[k] = [gammas[k]]
                
            for gamma in const_gammas:
                gammas[0].append(gamma)
                gammas[1].append(np.nan)
                gammas[2].append(np.nan)
            
        for k, key in enumerate(self._results):
            result[key] = self._set_assumptions(gammas[k], self.num_starts, is_gamma = True)
        
        self.num_starts = len(result['value'][0])
        
        # gammas = self.load_gammas(flags, flags.get('save_extension', ''))
        
        # const_gammas = kwargs.get('const_gammas', False)
        
        # if const_gammas and const_gammas != []:
            
        #     gammas = [[], [], []]
                
        #     for gamma in const_gammas:
        #         gammas[0].append(gamma)
        #         gammas[1].append(np.nan)
        #         gammas[2].append(np.nan)
        # result = {}
        # for k, key in enumerate(self._results):
        #     result[key] = self._set_assumptions(gammas[k], self.num_starts, is_gamma = True)
        
        # self.num_starts = len(result['value'][0])
        
        return result
    
    
    def load_gammas(self, flags, save_extension):
        
        gammas = np.zeros((3, self.model.grb.num_bins))
        
        # for i in range(self.model.grb.num_bins):
        for i in [7]:
            
            samples = self._load_gammas(i, flags)
            
            gammas[0, i] = np.nanpercentile(samples.df['G9'], 50)
            gammas[1, i] = np.nanpercentile(samples.df['G9'], 16) - gammas[0, i]
            gammas[2, i] = np.nanpercentile(samples.df['G9'], 84) - gammas[0, i]
            
        return gammas.tolist()
    
    
    def _load_gammas(self, i, flags):
        
        j = 0
        live_points = 400
        
        save_vars_temp = copy.deepcopy(self.save_vars_extend(i, j))
        
        save_vars_temp['results_path'] = '/Users/wistemar/Documents/GRB/Komrad/Results/180720598/lle'
        save_vars_temp['method_model'] = 'bsn_KRA'
        # save_vars_temp['method_model'] = 'jl_KRA'
        save_vars_temp['num_params'] = '6+0'
        save_vars_temp['start_info'] = 'start_{0}_live_{1}'.format(j, live_points)
        # extension_start = 'init_params_' + '_'.join(['{0}_{1}'.format(param, round(self.start_values[param][j], 3)) for param in self.start_values])
        # save_vars_temp['start_info'] = 'start_{0}_{1}'.format(j, extension_start)
        flags['converted'] = True
        
        try:
            samples = Samples.load(save_vars_temp, flags, flags['save_extension'])
        except FileNotFoundError as error:
            log.info(error)
            raise FileNotFoundError
        
        return samples
    
    
    @abstractmethod
    def fit(self, indices, loop_description, flags, **kwargs):
        
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        
        self.nai_min_rate, self.bgo_min_rate = kwargs.pop('nai_min_rate', 10), kwargs.pop('bgo_min_rate', 10)
        
        for index in tqdm(indices, desc = loop_description):
                
            mpi_comm.Barrier()
            
            # Extend save vars for current bin/start
            save_vars_extended = self.save_vars_extend(index, **kwargs)
            
            # Setup model for fit and see if fit already exists
            file_exists, path_fit_file = self._fit_setup(index, save_vars_extended, **kwargs)
            
            if file_exists and not flags['overwrite'] and mpi_rank == 0:
                # Loads fit
                log.info('Fit already exists, loads AR file. Count spectrum and walker path\n(if using bayesian MCMC) can be found in the Figures subfolder.')
                ar = load_analysis_results(path_fit_file)
                ar.display()
                    
            else:
                # performs fit
                try:
                    fit_results = self._perform_fit(index)
                except Exception as ex:
                    self._results['value'].at[index, 'Failed'] = True
                    log.warning('Fit failed. Exception: {0}. Arguments: {1}.'.format(type(ex).__name__, ex.args))
                    continue
                
                mpi_comm.Barrier()
                    
                if mpi_rank != 0:
                    continue
                
                # Gets the 3ML AR object
                ar = self._process_results(index, fit_results, save_vars_extended, path_fit_file, flags)
            
            if mpi_rank == 0:
                # Adds the fitted parameters to the results
                self._add(index, ar)
                
                # Gets and save samples from fit
                samples = self.get_samples(index, ar, save_vars_extended, flags)
                
                # Plots count spectra and walker path when applicable
                self._fit_plots(index, samples, flags, ar = ar, **kwargs)
                    
                self.save(save_vars_extended, flags)

        if flags['return_ar'] and mpi_rank == 0:
            return ar
        else:     
            return
        
        
    def fit_simulated_data(self, points, flags, **kwargs):
            
        for i in tqdm(points, desc = 'Points fitted'):
            
            save_vars_extended = self.simulated_save_vars(i)
            
            # Setup model for fit and see if fit already exists
            file_exists, path_fit_file = self.simulated_fit_setup(i, save_vars_extended, flags, **kwargs)
            
            if file_exists and not flags['overwrite']:
                # Loads fit
                log.info('Fit already exists, loads AR file. Count spectrum and walker path\n(if using bayesian MCMC) can be found in the Figures subfolder.')
                ar = load_analysis_results(path_fit_file)
                ar.display()
                
            else:
                # performs fit
                try:
                    fit_results = self._perform_fit(i, **flags)
                except Exception as ex:
                    self._results['value'].at[self._results['value'].iloc[[i]].index[0], 'Failed'] = True
                    log.warning('Fit failed. Exception: {0}. Arguments: {1}.'.format(type(ex).__name__, ex.args))
                    continue
                
                # Gets the 3ML AR object
                ar = self._process_results(i, fit_results, save_vars_extended, path_fit_file, flags)
                
            # Adds the fitted parameters to the results
            index = self._results['value'].index[2 * i + 1]
            self.simulated_add(index, ar)
            
            # Gets and save samples from fit
            _ = self.simulated_get_samples(index, ar, save_vars_extended, flags)
            
            # Plots count spectra and walker path when applicable
            if not (file_exists and not flags['overwrite']):
                self.simulated_fit_plots(i, flags, fit_results, save_vars_extended)
            
            # if self.method == 'bayesian':
            #     # Plots a corner plot for the fit
            #     flags['converted'] = False
            #     MAP_points = [ar.get_data_frame()['value'][k] for k, _ in enumerate(self.free_parameters)]
            #     self.corner(self.free_parameters, i, j, flags, MAP_points = MAP_points, save_vars = save_vars_extended, **kwargs)
                
            self._save_best_fit_params(save_vars_extended, flags)
        
        return
    
    
    # def simulated_save_vars(self, i) -> dict:
        
    #     save_vars_extended = self.save_vars_extend(i, None, fake = True)
        
    #     save_vars_extended['num_params'] = '{0}'.format(self.num_params)
    #     save_vars_extended['method_model'] = '{0}_{1}'.format(self.method_short, self.model.name)
        
    #     return save_vars_extended
    
    
    # def simulated_add(self, index, ar) -> None:
        
    #     for key, (k, param) in itertools.product(self._results.keys(), enumerate(self.free_parameters)):
    #         self._results[key].at[index, param] = ar._get_results_table(error_type = 'covariance', cl = 0.68, covariance = ar.covariance_matrix).frame[key].iloc[k]
        
    #     # Saves AIC value
    #     self._results['value'].at[index, 'AIC'] = round(ar._statistical_measures['AIC'], 2)
        
    #     for key in self._results:
    #         # Sets current fit to not have failed
    #         self._results[key].at[index, 'Failed'] = False
            
    #         # Saves the scale value to the temperature or Gamma, when applicable.
    #         if self.model.name == 'KRA':
    #             if self.num_paramsstartswith('5+0'):
    #                 self._results[key].at[index, 'Tob'] = self._results[key].at[index, 'scale']
    #             elif self.num_params.startsiwth('5+1'):
    #                 self._results[key].at[index, 'G'] = self._results[key].at[index, 'scale']
                    
    #     return
    
    
    # def simulated_get_samples(self, index, ar, save_vars_extended, flags):
        
    #     length = ar.samples.shape[1]
    #     short_length = 10000
    #     need_short_samples = length > short_length
        
    #     if self.method == 'bayesian':
    #         samples = self._simulated_get_samples(length, ar.samples.T, ar.log_probability, index, save_vars_extended, flags, need_short_samples)
    #     else:
    #         samples = self._simulated_get_samples(length, ar.samples.T, None, index, save_vars_extended, flags, need_short_samples)
        
    #     if need_short_samples:
            
    #         rng = np.random.default_rng()
    #         rows_to_keep = rng.choice(length, size = short_length, replace = False, shuffle = False)
            
    #         short_samples = copy.deepcopy(ar.samples)
    #         short_samples = short_samples[:, rows_to_keep]
            
    #         short_log_prob = copy.deepcopy(ar.log_probability)
    #         short_log_prob = short_log_prob[rows_to_keep]
            
    #         _ = self._simulated_get_samples(short_length, short_samples.T, short_log_prob, index, save_vars_extended, flags)
        
    #     return samples
    
    
    # def _simulated_get_samples(self, length, samples_df, log_prob, index, save_vars_extended, flags, need_short_samples = False):
        
    #     if self.method == 'bayesian':
    #         samples = Samples(length, self.columns + ['log_prob'])
    #         samples.df['log_prob'] = log_prob
    #     else:
    #         samples = Samples(length, self.columns)
            
    #     samples.df[self.free_parameters] = samples_df
        
    #     if self.model.name == 'KRA':
    #         if self.num_params.startswith('5+0'):
    #             samples.df['Tob'] = samples.df['scale']
    #         elif self.num_params.startswith('5+1'):
    #             for param in ['scale', 'G']:
    #                 samples.df[param] = self._results['value'].loc[index, param]          
    #     else:
    #         samples.df[self.fix_parameters] = self._results['value'].loc[index, self.fix_parameters]
            
    #     if need_short_samples:
    #         save_extension = '_long' + flags['save_extension']
    #     else:
    #         save_extension = flags['save_extension']
        
    #     flags['converted'] = False
    #     samples.save(save_vars_extended, flags, self.save_extension_setter(save_extension))
        
    #     return samples
    
    
    def simulated_fit_plots(self, i, flags, ar, save_vars_extended) -> None:
        
        if flags['view_count_spectra'] or flags.get('view_walker_path', False):
            log.info('Point: {0}'.format(i))
            min_rate = [self.nai_min_rate] * len([detec for detec in self.model.grb.detectors if detec.startswith('n')]) + [self.bgo_min_rate]
            data_colors = ['#FF0000', '#0000B7', '#009292', 'magenta', 'slateblue', 'orange']
            model_colors = ['lightcoral', 'darkblue', 'darkturquoise', 'darkmagenta', 'darkslateblue', 'darkorange']
            
            # If there is a bgo detector make it green/darkgreen
            if 'b0' in self.model.grb.detectors or 'b1' in self.model.grb.detectors:
                # If there's to many detectors
                if len(self.model.grb.detectors) - 1 > len(data_colors):
                    print('The number of detectors is unexpected')
                    return
                data_colors = data_colors[0:len(self.model.grb.detectors)-1]
                data_colors.append('#00E800')
                model_colors = model_colors[0:len(self.model.grb.detectors)-1]
                model_colors.append('darkgreen')
            
            fig, [ax, res_ax] = plt.subplots(nrows = 2, ncols = 1, sharex = True, height_ratios = [2.5, 1], figsize = (8, 5.25))
            display_spectrum_model_counts(ar, min_rate = min_rate, model_labels = ['_'] * self.model.grb.num_detectors, model_colors = model_colors[0:self.model.grb.num_detectors], data_colors = data_colors[0:self.model.grb.num_detectors], model_subplot = [ax, res_ax], show_data = True, show_legend = True, show_residuals = True)
            for ax_i in [ax, res_ax]:
                ax_i.grid(True, alpha = 0.5, which = 'both')
                for axis in list(ax_i.spines.keys()):
                    ax_i.spines[axis].set_linewidth(1.8)
                    ax_i.spines[axis].set_visible(True)
            ax.set_xlim([5e0, 5e4])
            
            add_save_vars = 'point_{0}_nai_{1}_bgo_{2}'.format(i, self.nai_min_rate, self.bgo_min_rate)
            path = '{results_path}/figures/spectra/count/count_{method_model}_params_{num_params}_{start_values}_0.pdf'.format(add_save_vars, **save_vars_extended)
            fig.savefig(path, transparent = True)
            
            plt.show()
            plt.close()
        return
        
        
    # def get_bins_starts(self, bins, starts):
    #     # If bins/starts are not defined, loop over all
    #     if bins is None or bins == []:
    #         bins = range(self.model.grb.num_bins)
    #     if starts == []:
    #         starts = range(self.num_starts)
    #     return bins, starts
        
        
    # @abstractmethod   
    # def save_vars_extend(self, i, j, extension):
        
    #     save_vars_extended = copy.deepcopy(self._save_vars)
    #     save_vars_extended['start_time'] = round(self.model.grb.detector_data[0].bins[i].start_time, 3)
    #     save_vars_extended['stop_time'] = round(self.model.grb.detector_data[0].bins[i].stop_time, 3)
    #     save_vars_extended['start'] = 'start_{0}'.format(j)
    #     save_vars_extended['i'] = i
    #     save_vars_extended['j'] = j
    #     save_vars_extended['start_info'] = '{0}_{1}'.format(save_vars_extended['start'], extension)
        
    #     return save_vars_extended
    
    
    def save_extension_setter(self, save_extension):
        return '_'.join(filter(None, [self.extension, save_extension]))
    
    
    # @abstractmethod
    # def _fit_setup(self, i, j, save_vars_extended, flags, **kwargs):
        
    #     if flags['is_fake_data']:
    #         try:
    #             path_fit_file = '{results_path}/fits_files/{0}/{method_model}_{start_values}{1}'.format(self.method, self.save_extension_setter(flags['save_extension']), **save_vars_extended)
    #         except:
    #             raise Exception('Problem with path fit file for fake data')
            
    #     else:
    #         path_fit_file = '{results_path}/fits_files/{0}/GRB{number}_{method_model}_params_{num_params}_{bin_method}_{bin_var}_time_{start_time}-{stop_time}_{start_info}{1}.fits'.format(self.method, self.save_extension_setter(flags['save_extension']), **save_vars_extended)
            
    #     file_exists = os.path.exists(path_fit_file)
        
    #     if flags['is_fake_data']:
    #         init_vals = list(self._results['value'].iloc[i][self.free_parameters])
    #         log.info('Point: {0}\nInitial: {1}'.format(i, ', '.join(['{0} = {1}'.format(param, init_vals[k]) for k, param in enumerate(self.parameters)])))

    #     else:
    #         log.info('Bin: {0}; Time {start_time}-{stop_time}'.format(i, **save_vars_extended))
    #         log.info('Start: {0}'.format(j))
        
    #     return file_exists, path_fit_file
    
    @abstractmethod
    def save_vars_extend(self, index):
        
        save_vars_extended = copy.deepcopy(self._save_vars)
        save_vars_extended['start_time'] = round(self.model.grb.detector_data[0].bins[index[0]].start_time, 3)
        save_vars_extended['stop_time'] = round(self.model.grb.detector_data[0].bins[index[0]].stop_time, 3)
        
        return save_vars_extended
    
    
    @abstractmethod
    def _perform_fit(self):
        return
    
    
    def _process_results(self, index, fit_results, save_vars_extended, path_fit_file, flags):
        
        ar = fit_results.results
        
        self._save_model_data(index, save_vars_extended)
        
        if flags['save_ar']:
            ar.write_to(path_fit_file, overwrite = True)
                
        return ar
    
    @abstractmethod
    def _save_model_data(self, i, save_vars_extended) -> None:
        
        detectors_long = self.model.grb.data[i].keys()
        
        for detector, detector_long in zip(self.model.grb.detectors, detectors_long):
            
            if detector.startswith('b'):
                min_rate = self.bgo_min_rate
            elif detector.startswith('lle'):
                min_rate = 0.
            else:
                min_rate = self.nai_min_rate
            
            new_data = self.model.grb.data[i][detector_long]._construct_counts_arrays(min_rate = min_rate, ratio_residuals = False)
            
            save_keys = ['new_observed_rate', 'new_chan_width', 'new_background_rate', 'new_observed_rate_err', 'new_background_rate_err', 'mean_energy', 'delta_energy', 'new_model_rate', 'residuals']
            save_dict = {}
            
            for key in save_keys:
                if key == 'delta_energy':
                    save_dict['delta_energy_low'] = np.array(new_data[key][0])
                    save_dict['delta_energy_high'] = np.array(new_data[key][1])
                else:
                    save_dict[key] = np.array(new_data[key])
            
            save_df = pd.DataFrame.from_dict(save_dict)
            path = self._model_data_path(detector, save_vars_extended)
            save_df.to_csv(path)
        
        return
    
    @abstractmethod
    def _model_data_path(self):
        return
    
    
    def _add(self, index, ar) -> None:
        
        self.add_to_df(index, ar)
        
        # Saves AIC value
        self._results['value'].at[index, 'AIC'] = round(ar._statistical_measures['AIC'], 2)
        
        for key in self._results:
            # Sets current fit to not have failed
            self._results[key].at[index, 'Failed'] = False
            
            # Saves the scale value to the temperature or Gamma, when applicable.
            if self.model.name == 'KRA':
                if self.num_params.startswith('5+0'):
                    self._results[key].at[index, 'Tob'] = self._results[key].at[index, 'scale']
                elif self.num_params.startswith('5+1'):
                    self._results[key].at[index, 'G'] = self._results[key].at[index, 'scale']
                    
        return
    
    
    def get_samples(self, index, ar, save_vars_extended, flags):
        
        length = ar.samples.shape[1]
        short_length = 10000
        need_short_samples = length > short_length
        
        try:
            samples = self._get_samples(length, ar.samples.T, ar.log_probability, index, save_vars_extended, flags, need_short_samples)
        except AttributeError:
            samples = self._get_samples(length, ar.samples.T, np.nan, index, save_vars_extended, flags, need_short_samples)
        
        if need_short_samples:
            
            rng = np.random.default_rng()
            rows_to_keep = rng.choice(length, size = short_length, replace = False, shuffle = False)
            
            short_samples = copy.deepcopy(ar.samples)
            short_samples = short_samples[:, rows_to_keep]
            
            short_log_prob = copy.deepcopy(ar.log_probability)
            short_log_prob = short_log_prob[rows_to_keep]
            
            _ = self._get_samples(short_length, short_samples.T, short_log_prob, index, save_vars_extended, flags)
        
        return samples
    
    
    def _get_samples(self, length, samples_df, log_prob, index, save_vars_extended, flags, need_short_samples = False):
        
        samples = Samples(length, self.columns + ['log_prob'])
        samples.df['log_prob'] = log_prob
            
        samples.df[self.free_parameters] = samples_df
        samples.df[self.fix_parameters] = self._results['value'].loc[index, self.fix_parameters]
        
        if self.model.name == 'KRA':
            if self.num_params.startswith('5+0'):
                samples.df['Tob'] = samples.df['scale']
            elif self.num_params.startswith('5+1'):
                samples.df['G'] = samples.df['scale']  
            
        if need_short_samples:
            save_extension = '_long' + flags['save_extension']
        else:
            save_extension = flags['save_extension']
        
        flags['converted'] = False
        samples.save(save_vars_extended, flags, self.save_extension_setter(save_extension))
        
        return samples
    
    
    @abstractmethod
    def _fit_plots(self, index, flags, **kwargs) -> None:
        
        flags['data'] = True
        flags['residuals'] = True
        flags['mode'] = 'count'
        self.spectra([index[0]], [index[1]], flags, **kwargs)
        
        return
    
    
    @abstractmethod
    def _save_best_fit_params(self, flags) -> None:
        return
    
    
    def transform(self, flags, stat_type = 'Fit', bins = None, starts = [0], **kwargs) -> None:
        
        bins, starts = self.get_bins_starts(bins, starts)
        
        if self.num_params.startswith('5+1'):
            t_var_name = 'tvar_pre'
        else:
            t_var_name = 'tvar'
        
        try: 
            self.assumption_params.remove(t_var_name)
        except ValueError:
            log.info('tvar name already removed')
            
        
        for i in tqdm(bins, desc = 'Bins'):
            for j in tqdm(starts, desc = 'Starts'):
            
                if self._results['value'].loc[(i, j, stat_type), 'Failed']:
                    continue
                
                save_vars_extended = self.save_vars_extend(i, j, **kwargs)
                
                if not flags['errors']:
                    convert.convert(self, self.model, i, j = j, stat_type = stat_type, freeze_gamma = self.model.freeze_gamma)
                    
                else:
                    samples = Samples.load(save_vars_extended, flags, self.save_extension_setter(flags['save_extension']), stat_type)
                    samples.num_params = self.num_params
                    if not flags['converted']:
                        
                        if not np.isnan(self._results['value'].at[(i, j, 'Fit'), t_var_name]) and np.isnan(samples.df[t_var_name][0]):
                            t_vars = self.get_gaussian_tvar(t_var_name, i, j, samples)
                            samples.df[t_var_name] = t_vars
                            # In order to not re-generate t_var every time
                            samples.save(save_vars_extended, flags, self.save_extension_setter(flags['save_extension']), stat_type = stat_type)
                        
                        # gamma_samples = self._load_gammas(i, flags_converted)
                        # samples.df['G'] = gamma_samples.df['G9']
                        
                        for k in tqdm(samples.df.index, desc = 'Converting samples'):
                            samples.df.loc[(k), self.assumption_params] = self.get_params_value(self.assumption_params, i, j, stat_type = stat_type, **kwargs)
                            samples.assumption_params = self.assumption_params
                            convert.convert(samples, self.model, k, freeze_gamma = self.model.freeze_gamma)
                        
                        flags_converted = flags.copy()
                        flags_converted['converted'] = True
                        samples.save(save_vars_extended, flags_converted, self.save_extension_setter(flags['save_extension']), stat_type = stat_type)
                        
                    convert.convert(self, self.model, i, j = j, samples_for_errors = samples.df, stat_type = stat_type, freeze_gamma = self.model.freeze_gamma)
                    
                self.save(save_vars_extended, flags)
        
        return
    
    
    def get_gaussian_tvar(self, t_var_name, i, j, samples):
    
        mu = self._results['value'].loc[(i, j, 'Fit'), t_var_name]
        sigma = self._results['positive_error'].loc[(i, j, 'Fit'), t_var_name]
        lower_limit = 0
        upper_limit = 2 * mu
        
        a, b = (lower_limit - mu) / sigma, (upper_limit - mu) / sigma
        
        distribution = truncnorm(a, b, loc = mu, scale = sigma)
        random_variates = distribution.rvs(size = len(samples.df))    
    
        return random_variates
    
    
    def get_params_value(self, params, i, j, **kwargs):
        stat_type = kwargs.get('stat_type', 'Fit')
        key = kwargs.get('key', 'value')
        
        if not isinstance(params, list):
            values = self.get_value(params, i, j, stat_type, key)
        else:
            for param in params:
                values = [self.get_value(param, i, j, stat_type, key) for param in params]
        
        return values
    
    
    def store_params_value(self, params, values, i, j, **kwargs) -> None:
        stat_type = kwargs.get('stat_type', 'Fit')
        key = kwargs.get('key', 'value')
        
        if not isinstance(values, list):
            self.store_value(params, i, j, stat_type, key, values)
        else:
            for param, value in zip(params, values):
                self.store_value(param, i, j, stat_type, key, value)
                
        return
    
    
    def get_value(self, param, i, j, stat_type, key) -> None:
        return self._results[key].at[(i, j, stat_type), param]
    
    
    def store_value(self, param, i, j, stat_type, key, value) -> None:
        self._results[key].at[(i, j, stat_type), param] = value
        return
    
    
    def get_values(self, param, key, bins = [], starts = []):
        bins, starts = self.get_bins_starts(bins, starts)
        if self.save_vars['method_model'] == 'jl_Band' or self.save_vars['method_model'] == 'jl_dsbpl' or self.save_vars['method_model'] == 'jl_DoubleSmoothlyBrokenPowerlaw':
            starts = [0]
            
        param_values = np.zeros((len(starts), len(bins)))
        for (k, i), (l, j) in itertools.product(enumerate(bins), enumerate(starts)):
            param_values[l, k] = self.get_value(param, i, j, 'Fit', key)
        
        return param_values
    
    
    def round_number(self, param):
        if param == 'rph10':
            round_number = -9
        elif param == 'zeta':
            round_number = -3
        elif param == 'R' or param == 'G4' or param == 'G8' or param == 'G10':
            round_number = 0
        elif param == 'tt' or param == 'tau':
            round_number = 1
        elif param == 'tu' or param == 'tuk' or param == 'z':
            round_number = 3
        else:
            round_number = 2
        return round_number
    
    
    def corner(self, params, i, j, flags, **kwargs) -> None:
            
        params = self.param_abb_list(params)
        
        if flags['view_corner']:
            log.info('Bin: {0}'.format(i))
        
        save_vars_extended = self.save_vars_extend(i, j, **kwargs)
        
        corner_plot = Corner(self, save_vars_extended, params, i, j)
        corner_plot.plot(flags, **kwargs)
        
        return
    
    
    def spectra(self, bins, starts, flags, **kwargs) -> None:
        
        if 'min_rate' in kwargs:
            self.nai_min_rate = kwargs.get('min_rate')
            self.bgo_min_rate = kwargs.pop('min_rate')
            
        else:
            self.nai_min_rate = kwargs.pop('nai_min_rate', 30)
            self.bgo_min_rate = kwargs.pop('bgo_min_rate', 12)
        
        bins, starts = self.get_bins_starts(bins, starts)
        
        mode = flags.get('mode', None)
        spectrum_class_dict = {'resolved' : ResolvedSpectra, 'evolution' : SpectralEvolution, 'count' : CountSpectra}
        if mode not in spectrum_class_dict:
            raise ValueError('Mode: {0} is not supported!'.format(mode))
        
        spectra = spectrum_class_dict[mode](self, bins, starts, flags, mode, **kwargs)
        spectra.plot(**kwargs)
        
        return
    
    
    @staticmethod
    def AIC(bins, starts, flags, AIC_thr, y_upper_lim, *best_fits_s) -> None:
        
        # bins, starts = best_fits_s[0].get_bins_starts(bins, starts)
        
        AIC_plot = AIC(bins, starts, *best_fits_s)
        AIC_plot.plot(flags, AIC_thr, y_upper_lim)
        
        return
    
    
    def param_evolution(self, bins, starts, params, **kwargs) -> None:
        
        bins, starts = self.get_bins_starts(bins, starts)
        
        param_evo_plot = ParamEvo(self, bins, starts)
        param_evo_plot.plot(params, **kwargs)
        
        return
    
    
    def param_abb_list(self, params):
        
        if self.model.name == 'KRA':
        
            KRA_params, RMS_params, burst_params = constants.constants(['KRA_PARAMS', 'RMS_PARAMS', 'BURST_PARAMS{0}'.format(self.num_params[2])])
            
            if 'KRA' in params:
                pos = params.index('KRA')
                params[pos:pos+1] = KRA_params
                
            if 'RMS' in params:
                pos = params.index('RMS')
                params[pos:pos+1] = RMS_params
                
            if 'Burst' in params:
                pos = params.index('Burst')
                params[pos:pos+1] = burst_params
            
        flux_params = constants.constants(['FLUX_PARAMS'])
        
        if 'Flux' in params:
            pos = params.index('Flux')
            params[pos:pos+1] = flux_params
            
        return params


    def _set_assumptions(self, assumptions, num_starts, recursive = False, is_gamma = False):
            
        # Only number
        if isinstance(assumptions, (float, int)):
            result = np.full((self.model.grb.num_bins, num_starts), assumptions)
            
        # Only list with numbers
        elif all(isinstance(assumption_val, (float, int)) for assumption_val in assumptions):
            if len(assumptions) == 1:
                result = np.full((num_starts, self.model.grb.num_bins), assumptions * self.model.grb.num_bins).T
            elif len(assumptions) != self.model.grb.num_bins:
                if recursive:
                    raise ValueError('A 1D list needs to be as long as the number of bins.')
                else:
                    raise ValueError('A 1D list needs to be as long as the number of bins. For specifying different start values per bin, make a 2D list.')
            else:
                result = np.full((num_starts, self.model.grb.num_bins), assumptions).T
            
        # 2D lists
        else:
            if recursive:
                raise ValueError('3D lists are not supported for assignment.')
                
            results = []
            
            if len(assumptions) != num_starts and not is_gamma:
                assumptions = (assumptions * (int(num_starts / len(assumptions)) + 1))[0:num_starts]
            
            for assumption_val in assumptions:
                results.append(self._set_assumptions(assumption_val, num_starts = 1, recursive = True, is_gamma = is_gamma))
                
            if self.method == 'joint_likelihood':
                if is_gamma:
                    result = np.concatenate(results[0:1] * num_starts, axis = 1)
                else:
                    result = np.concatenate(results, axis = 1)
            else:
                result = np.concatenate(results, axis = 1)
        
        return result
    
    
    def set_assumptions(self, assumptions, negative_errors, positive_errors) -> None:
        
        self.assumption_params = list(assumptions.keys())
        # Since t_var now has a distribution, remove it from assumed
        # self.assumption_params.remove('tvar')
        
        values = dict(zip(self._results.keys(), [assumptions, negative_errors, positive_errors]))
        for key, param in itertools.product(values, assumptions):
            results = self._set_assumptions(values[key].get(param, 0), self.num_starts)
            
            for i, j in itertools.product(range(self.model.grb.num_bins), range(self.num_starts)):
                self.store_params_value(param, results[i, j], i, j, stat_type = 'Fit', key = key)

        return
    
    
    def add_assumption_row(self, j, assumptions, negative_errors, positive_errors, rows_to_add) -> None:
        
        values = dict(zip(self._results.keys(), [assumptions, negative_errors, positive_errors]))
        
        for key, param in itertools.product(values, assumptions):
            if rows_to_add != 0:
                results = self._set_assumptions(values[key].get(param, 0), rows_to_add)
            
            for i, k in itertools.product(range(self.model.grb.num_bins), range(rows_to_add)):
                self._results[key].loc[(i, j, 'Fit{0}'.format(k + 1))] = self._results[key].loc[(i, j, 'Fit')]
                self.store_params_value(param, results[i, k], i, j, stat_type = 'Fit{0}'.format(k + 1), key = key)
                
            self._results[key] = self._results[key].drop(axis = 0, level = 'Type', index = self._results[key].index.levels[2][rows_to_add + 1:None]).sort_index()
        
        return
    

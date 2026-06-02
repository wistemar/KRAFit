import astropy.units as u
import copy
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from threeML import TemplateModel, TemplateModelFactory
from threeML.utils.progress_bar import tqdm, trange

from KRAFit.Convert import temperature_path
from KRAFit.Models.Model import ImportModel

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from KRAFit.Setup.grb_setup import GRB

class KRA(ImportModel):
    
    def __init__(self, mode: str, parent_file: str, name: str, grb: 'GRB', view_spectra: bool = False, freeze_gamma: bool = False, multi_powerlaw: bool = False, **kwargs) -> None:
        
        self.mode = mode
        self.parent_file = parent_file
        self.plot = view_spectra
        self.freeze_gamma = freeze_gamma
        self.multi_powerlaw = multi_powerlaw
        
        self._count_parent_file()
        
        super().__init__(name, grb, None, **kwargs)
        
        return
    
    
    def __str__(self) -> str:
        num_params = self.mode[1]
        if self.mode[0] == 'n':
            save_load = 'new'
        elif self.mode[0] == 'l':
            save_load = 'loaded'
            
        return 'Using {0} parameters with a {1} {2} model.'.format(num_params, save_load, self.__class__.__name__)
    
    
    def __repr__(self) -> str:
        return '{0}(mode = {1}, parent_file = \'name.txt\', name = {2}, grb = {3!r})'.format(
            self.__class__.__name__, self.mode, self.name, self.grb)
    
    
    def _count_parent_file(self) -> None:
        # Counts number of spectra in parent file
        with open(self.parent_file, 'r') as file:
            self.n_spectra = 0
            for i, line in enumerate(file.readlines()):
               self.n_spectra += 1
               if i == 0:
                   _, _, _, self.n_bins = self._read_spectrum_file(line[:-1])
        
        return


    def import_model(self, tr_nBins: int = 10, tr_min: float = 0.01, tr_max: float = 0.25, nBins_intermittence: int = 1, **kwargs):
        
        if self.multi_powerlaw:
            extra_pl = 'MPL'
        elif self.add_powerlaw:
            extra_pl = 'APL'
        else:
            extra_pl = ''
        
        if 'dynamical' in self.parent_file:
            self._alt_name = '{0}{1}{2}'.format(self.name, self.mode[-1], extra_pl)
            self.spectral_shape_name = self.spectral_shape_name + '{0}{1}'.format(self.mode[-1], extra_pl)
        else:
            self._alt_name = '{0}old{1}{2}'.format(self.name, self.mode[-1], extra_pl)
            self.spectral_shape_name = self.spectral_shape_name + 'old{0}{1}'.format(self.mode[-1], extra_pl)
        
        if self.mode[0] == 'l':
            file_name = '{0}/energies/energies_{1}.npy'.format(self.grb.results_path, self.mode[1])
            self.energies = np.load(file_name)
        elif self.mode[0] == 'n':
            self._create_TemplateModel(tr_nBins, tr_min, tr_max, nBins_intermittence)
        else:
            raise ValueError('Not a valid value for mode')

        KRA_template_model = TemplateModel(self._alt_name)
        
        KRA_template_model.K.min_value     = 1e-5
        KRA_template_model.K.max_value     = 10000.
        
        if self.mode[-1] == '6':
            if self.freeze_gamma:
                KRA_template_model.scale.free = False
            else:
                KRA_template_model.scale.min_value = 1e0
                KRA_template_model.scale.max_value = 1e3
            
        elif self.mode[-1] == '5':
            KRA_template_model.scale.min_value = 1e-4
            KRA_template_model.scale.max_value = 1e4
            
        else:
            None
            
        return KRA_template_model
    
    
    def _create_TemplateModel(self, tr_nBins: int, tr_min: float, tr_max: float, nBins_intermittence: int) -> None:
        # Read spectral files
        N_nu, read_params, energies_original = self._read_spectral_files()
        
        index_energy_is_one = np.argwhere(energies_original == 0.6)[0, 0]
        
        # Setup the energy array
        self._setup_energies_array(energies_original, read_params, tr_min, tr_max)
        energies_im = self.energies[0::nBins_intermittence]
        
        # Setup param grid
        params = self._setup_param_grid(read_params, tr_min, tr_max, tr_nBins)
            
        # Create TemplateModel
        energies_temp = energies_im * u.keV
        temp_factory = TemplateModelFactory(self._alt_name, self._alt_name, energies_temp, params)
        for param in self.param_grid:
            temp_factory.define_parameter_grid(param, self.param_grid[param])
        
        # Add spectra to the template model
        temp_factory = self._add_spectra(N_nu, read_params, energies_original, index_energy_is_one, temp_factory, nBins_intermittence)
        
        # Save energy array and TemplateModel
        file_name = '{0}/energies/energies_{1}'.format(self.grb.results_path, self.mode[1])
        np.save(file_name, self.energies)
        temp_factory.save_data(overwrite = True)

        return
    

    def _read_spectral_files(self):
        
        # Parameters read from the spectral files
        read_params = {
            'tt' : np.zeros((self.n_spectra, )),
            'R'  : np.zeros((self.n_spectra, )),
            'yr' : np.zeros((self.n_spectra, )),
            }
        
        # Creates array that stores the photon count
        N_nu = np.zeros((self.n_spectra, self.n_bins))
        
        # Reads all spectral files   
        with open(self.parent_file, 'r') as file:
        
            for i, line in enumerate(tqdm(file.readlines(), desc = 'Reading spectra: ')):
                read_param, energies_original, spectrum, _ = self._read_spectrum_file(line[:-1])
     
                # Adds the spectrum to the photon count array       
                N_nu[i, ] = spectrum
                
                # Adds the parameters for each spectra
                for param in read_params:
                    read_params[param][i] = read_param[param]  
                    
        return N_nu, read_params, energies_original
    
    
    def _read_spectrum_file(self, file_name: str):
        header_line = True
        count = 0
        k = 0
        with open(file_name, 'r') as file:
            for line in file.readlines():
                line_info = line.split(' ')
                if header_line:
                    if '%' in line_info[0]:
                        count += 1
                        if count == 6:
                            header_line = False
                        continue
                    elif line_info[0] == 'nBins':
                        n_bins = int(line_info[2])
                        energies = np.zeros(n_bins)
                        spectrum  = np.zeros(n_bins)
                    elif line_info[0] == 'Gamma':
                        pass
                    elif line_info[0] == 'radius':
                        pass
                    elif line_info[0] == 'tau':
                        tau = float(line_info[2])
                    elif line_info[0] == 'theta_r' or line_info[0] == 'theta_RMS':
                        theta_r = float(line_info[2])
                    elif line_info[0] == 'R_theta':
                        R = float(line_info[2])
                    elif line_info[0] == 'y_RMS' or line_info[0] == 'y_r':
                        yr = float(line_info[2])
                    elif  line_info[0] == '<eps>':
                        pass
                    elif  line_info[0] == 'theta_C':
                        pass
                    elif  line_info[0] == 'varphi':
                        pass
                    elif line_info[0] == 'N_u':
                        pass
                else:
                    energies[k] = float(line_info[1])
                    spectrum[k]  = float(line_info[2])
                    k += 1

        read_param = {
            'tt' : tau * theta_r,
            'R'  : R                 ,
            'yr' : yr                ,
            }

        return read_param, energies, spectrum, n_bins
    
    
    def _setup_energies_array(self, energies_original: npt.NDArray[np.float64], read_params: dict, tr_min: float, tr_max: float):
        if self.mode == 'n5':
            self.energies = energies_original
        elif self.mode == 'n6':
            # Spectra moves to observed except for gamma so observed / gamma since we want scale to be gamma
            # Thus we need to rescale the energies array    
            e_min = np.min(energies_original) * temperature_path.norm_to_gamma_frame(self.grb.z, np.max(read_params['tt']), np.max(read_params['R']), tr_min)
            e_max = np.max(energies_original) * temperature_path.norm_to_gamma_frame(self.grb.z, np.min(read_params['tt']), np.min(read_params['R']), tr_max)
            # The increment should be evenly logspaced, i.e., energies[i+1]/energies[i] should be the same for all i
            # Here, we take the mean, because it turns out there are slight (as in very tiny) variations across the grid
            increment = np.mean(energies_original[1:] / energies_original[:-1])
            self.energies = np.array([e_min])
            while self.energies[-1] < e_max:
                self.energies = np.append(self.energies, self.energies[-1] * increment)
        return
    
    
    def _setup_param_grid(self, read_params, tr_min, tr_max, tr_nBins):
        self.param_grid = dict.fromkeys(read_params.keys())
        for param in read_params:
            self.param_grid[param] = np.unique(read_params[param])
        if self.mode == 'n6':
            self.param_grid['tr'] = np.linspace(tr_min, tr_max, tr_nBins)
            
        if self.multi_powerlaw:
            self.param_grid['p'] = np.linspace(-8, -2, 10) ## NEW
        
        params = list(self.param_grid.keys())
        
        return params
    
    
    def _add_spectra(self, N_nu: npt.NDArray[np.float64], read_params: dict, energies_original: npt.NDArray[np.float64], index_energy_is_one: int, temp_factory, nBins_intermittence: int):
        
        errors = [] 
        if self.plot:
            fig, ax = plt.subplots(figsize = (8, 5))
        
        for i in trange(self.n_spectra, desc = 'Importing spectra: '):
            for tr in self.param_grid.get('tr', [1]):
                
                N_nu_temp = np.zeros((len(self.energies), ))
                
                if self.mode == 'n5':
                    cm_change = 1
                elif self.mode == 'n6':
                    cm_change = temperature_path.norm_to_gamma_frame(self.grb.z, read_params['tt'][i], read_params['R'][i], tr)
                    
                lowest_energy_obs = energies_original[0] * cm_change
                index_start = np.abs(np.log10(self.energies) - np.log10(lowest_energy_obs)).argmin()
  
                # This is to see how large of an error this way of shifting the energy grid induces
                errors.append(np.max([lowest_energy_obs / self.energies[index_start], self.energies[index_start] / lowest_energy_obs]))

                #### WHY DO WE DO THE SECOND NORMALIZATION??
                N_nu_temp[index_start : index_start + self.n_bins] = N_nu[i, :] / N_nu[i, index_energy_is_one]
                
                if self.mode == 'n5':
                    N_nu_norm = N_nu_temp
                elif self.mode == 'n6':
                    index = np.argmax(N_nu_temp[index_start : index_start + self.n_bins] * energies_original**2)
                    N_nu_norm = N_nu_temp / N_nu_temp[index + index_start]
                
                
                for p in self.param_grid.get('p', [1]):
                    
                    N_nu_new = copy.deepcopy(N_nu_norm)
                    
                    if self.multi_powerlaw:
                        max_index = np.argmax(N_nu_new * self.energies**2)
                        
                        for k in np.linspace(max_index, len(self.energies), len(self.energies) - max_index + 1, dtype = int):
                            y_2 = np.log(N_nu_new[k + 1] * self.energies[k + 1]**2)
                            y_1 = np.log(N_nu_new[k] * self.energies[k]**2)
                            x_2 = np.log(self.energies[k + 1])
                            x_1 = np.log(self.energies[k])
                            
                            y_diff = y_2 - y_1
                            x_diff = x_2 - x_1
                            y_der = y_diff / x_diff
                            
                            if y_der < (p + 2):
                                slope_index = k
                                break
    
                        K = N_nu_new[slope_index] * self.energies[slope_index]**(-p)
                        N_nu_new[slope_index:] = K * self.energies[slope_index:]**(p)
                    
                    N_nu_new[N_nu_new == 0] = 1e-50
                    N_nu_new = N_nu_new[0 :: nBins_intermittence]
                    
                    if self.mode == 'n6':
                        parameters = {'tt' : read_params['tt'][i], 'R' : read_params['R'][i], 'yr' : read_params['yr'][i], 'tr' : tr}
                        
                    elif self.mode == 'n5':
                        parameters = {'tt' : read_params['tt'][i], 'R' : read_params['R'][i], 'yr' : read_params['yr'][i]}
                        
                    if self.multi_powerlaw:
                        parameters['p'] = p
                    
                    temp_factory.add_interpolation_data(N_nu_new, **parameters)
                    
                    if self.plot:
                        if i % 10 == 0:
                            ax.plot(self.energies, N_nu_new, color = (i * 1.0 / (1.3 * self.n_spectra), 0.1, 0.1))
        
        if self.plot:
            ax.set_xlim([1e-2, max(self.energies) * 2])
            ax.set_ylim([1e-3, 1e6])
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        return temp_factory
    
    
    def setup_start_values(self, K_start: list, scale_start: list, gamma_start: list, tt_start: list, R_start: list, yr_start: list, tr_start: list, p_start: list, index_start: list, xc_start: list) -> dict:
        
        start_values = {}
        
        for param in self.parameters:
            for start_params in start_values:
                if start_params:
                    pass
        
        start_values['K'] = K_start
        
        if self.mode[-1] == '5':
            start_values['scale'] = scale_start
        elif self.mode[-1] == '6':
            start_values['scale'] = gamma_start
        else:
            raise ValueError('Invalid mode.')
            
        start_values['tt'] = tt_start
        start_values['R']  = R_start
        start_values['yr'] = yr_start
        
        if self.mode[-1] == '6':
            start_values['tr'] = tr_start
        
        if self.add_powerlaw:
            start_values['p'] = p_start
            
        return start_values
    
    
    def setup_priors(self, K_prior, scale_prior, gamma_prior, tt_prior, R_prior, yr_prior, tr_prior, p_prior) -> None:
        
        self.spectra['K'].prior = K_prior
        self.spectra['tt'].prior = tt_prior
        self.spectra['R'].prior  = R_prior
        self.spectra['yr'].prior = yr_prior
        
        if self.mode[-1] == '5':
            self.spectra['scale'].prior = scale_prior
        elif self.mode[-1] == '6':
            self.spectra['scale'].prior = gamma_prior
            self.spectra['tr'].prior    = tr_prior
        else:
            raise ValueError('Invalid mode.')
        
        if self.multi_powerlaw:
            self.spectra['p'].prior = p_prior
        
        return
    

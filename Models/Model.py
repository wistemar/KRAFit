from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from threeML import Model, PointSource, SpectralComponent

from KRAFit.Misc.custom_astromodels import Powerlaw, Exponential_cutoff


class ImportModel(ABC):
    
    def __init__(self, name: str, grb, energies: npt.NDArray[np.float64], add_powerlaw: bool = False, exp_cutoff: bool = False, **kwargs) -> None:
        self.grb = grb
        self.name = name
        self.energies = energies
        self.spectral_shape_name = '{0}.spectrum.main.{1}'.format(self.grb.name, self.name)
        self.add_powerlaw = add_powerlaw
        self.exp_cutoff = exp_cutoff
        
        template_model = self.import_model(**kwargs)
        
        if self.add_powerlaw:
            # component1 = SpectralComponent('main', shape = template_model)
            # component2 = SpectralComponent('afterglow', shape = Powerlaw())
            # self.model = Model(PointSource(grb.name, grb.ra, grb.dec, components = [component1, component2]))
            # self.spectra = self.model['{0}'.format(self.grb.name)]
            # self.kra_spectra = self.model[self.spectral_shape_name]
            # self.powerlaw_spectra = self.model['{0}'.format(self.grb.name)]['spectrum']['afterglow']['Powerlaw']
            
            self.model = Model(PointSource(grb.name, grb.ra, grb.dec, spectral_shape = template_model + Powerlaw()))
            
            self.spectral_shape_name = '{0}.spectrum.main.composite'.format(self.grb.name)
            self.spectra = self.model[self.spectral_shape_name]
            

        elif self.exp_cutoff:
            self.model = Model(PointSource(grb.name, grb.ra, grb.dec, spectral_shape = template_model * Exponential_cutoff()))

            self.spectral_shape_name = '{0}.spectrum.main.composite'.format(self.grb.name)
            self.spectra = self.model[self.spectral_shape_name]
            self.spectra['K_2'].free = False

        else:
            self.model = Model(PointSource(grb.name, grb.ra, grb.dec, spectral_shape = template_model))
            self.spectra = self.model[self.spectral_shape_name]
            
        self._parameters = [param[:-2] if param != 'K_2' else param for param in self.spectra.parameters.keys()]
        self._free_parameters = [param[:-2] if param != 'K_2' else param for param in self.spectra.free_parameters.keys()]
        self._fix_parameters = [param for param in self._parameters if param not in self._free_parameters]
        
        self._parameters.remove('redshift')
        self._fix_parameters.remove('redshift')
        
        return
    
    
    @abstractmethod
    def import_model():
        pass
    
    
    @property
    def parameters(self) -> list:
        return self._parameters
    
    
    @property
    def free_parameters(self) -> list:
        params = list(self.spectra.parameters.keys())
            
        if params[0].endswith('_1'):
            for k, param in enumerate(params):
                if param == 'K_2':
                    pass
                else:
                    params[k] = param[:-2]
                    
        if 'redshift' in params:
            params.remove('redshift')
        
        return params
    
    
    @property
    def fix_parameters(self) -> list:
        params = list(self.spectra.parameters.keys())
            
        if params[0].endswith('_1'):
            for k, param in enumerate(params):
                if param == 'K_2':
                    pass
                else:
                    params[k] = param[:-2]
                    
        if 'redshift' in params:
            params.remove('redshift')
        
        return params
    
    
    def set_parameters(self, best_fits, i, j, stat_type) -> None:
        
        if best_fits.method == 'samples':
            self._set_parameters(best_fits.df[self.parameters].loc[(i), :])
        else:
            self._set_parameters(best_fits._results['value'][self.parameters].loc[(i, j, stat_type), :])
        
        return
    
    
    def _set_parameters(self, values) -> None:
        values = list(values)
        for k, param in enumerate(self.parameters):
            self.spectra[param].value = values[k]
        
        return


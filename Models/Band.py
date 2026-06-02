from threeML import Band

from KRAFit.Models.Model import ImportModel

class Band_import(ImportModel):
    
    def __init__(self, name, grb, energies, **kwargs):
        
        super().__init__(name, grb, energies, **kwargs)
        
        return
    
    
    def __str__(self):            
        return 'A {0} model.'.format(self.__class__.__name__.split('_')[0])
    
    
    def __repr__(self):
        return '{0}(name = {1}, grb = {2!r})'.format(self.__class__.__name__, self.name, self.grb)
    
    
    def import_model(self, **kwargs):
        
        band = Band()
        band.K               = 1
        band.alpha           = -0.8
        band.alpha.min_value = -3.
        band.beta            = -3.
        band.beta.min_value  = -10.
        band.xp              = 300.
        
        return band
        
import csv
import os
import re
from datetime import datetime
from glob import glob
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import bs4
import matplotlib.pyplot as plt
import requests
from mpi4py import MPI
from threeML import FermiGBMBurstCatalog, TimeSeriesBuilder, DataList, silence_warnings, activate_warnings, silence_logs, activate_logs
from threeML.io.logging import setup_logger

log = setup_logger(__name__)

from KRAFit.Misc import misc
from KRAFit.Plots.Plot import Plot


class GRB:
    
    def __init__(self, grb_number, SNR, p0, dt, z, data_path, results_path, bin_method, detector_method, bg_intervals, custom_start_stop, overwrite = False):
        self.number = grb_number
        self.name = 'GRB{number}'.format(number = grb_number)
        self.short_name = 'GRB{number}'.format(number = grb_number[0:6])
        self.SNR = SNR
        self.p0 = p0
        self.dt = dt
        self.z = z
        self.data_path = data_path
        self.results_path = results_path
        self.bin_method = bin_method
        self.detector_method = detector_method
        self.bg_intervals = bg_intervals
        self.custom_start_stop = custom_start_stop
        self.silence = True
        
        return
    
    
    def __str__(self):
        return 'A {0} object for {1} with {2} binning method (var = {3}) at {4}-{5} s, using detectors: {6}'.format(
            self.__class__.__name__, self.short_name, *self.bin_method, self.start, self.stop, ', '.join(self.detectors))
    
    
    def __repr__(self):
        return '{0}({1}, {2}, {3}, {4}, {5}, data_path, results_path, {6}, {7}, {8}, {9})'.format(
            self.__class__.__name__, self.number, self.SNR, self.p0, self.dt, self.z, self.bin_method, self.detector_method, self.bg_intervals, self.custom_start_stop)
        
        
    def create_folders(self) -> None:
        
        # Creates folder that holds data
        data_folder = '{0}/{1}'.format(self.data_path, self.number)
        misc.create_folder(data_folder)
        
        # Creates all results folders
        misc.create_folder(self.results_path)
        folders = {'figures'          : ['light_curves', 'param_evolution', 'walker_paths', 'spectra', 'corner_plots', 'AIC'],
                   'time_bins'        : [],
                   'best_fit_params'  : ['joint_likelihood', 'bayesian'],
                   'fits_files'       : ['joint_likelihood', 'bayesian'],
                   'samples'          : [],
                   'model_data'       : [],
                   'simulated_data'   : [],
                   'setups'           : [],
                   'energies'         : [],
                   'figures/spectra'  : ['resolved', 'evolution', 'count'],}
        
        for folder in folders:
            misc.create_folder('{0}/{1}'.format(self.results_path, folder))
            for sub_folder in folders[folder]:
                misc.create_folder('{0}/{1}/{2}'.format(self.results_path, folder, sub_folder))
                
        return
    
    
    def get_info(self, load: bool = False) -> None:
        
        try:
            self.burst_info = misc.load_obj('{0}/burst_info/{1}.pkl'.format(self.data_path, self.name))
        except FileNotFoundError:
            misc.create_folder('{0}/burst_info/'.format(self.data_path))
            log.info('No burst_info file exists, downloads.')
            gbm_catalog = FermiGBMBurstCatalog()
            gbm_catalog.query_sources(self.name)
            self.burst_info   = gbm_catalog.get_detector_information()
            misc.save_obj(self.burst_info, '{0}/burst_info/{1}.pkl'.format(self.data_path, self.name))
            
        self.trigger_name = self.burst_info.get(self.name).get('trigger')
        self.ra           = self.burst_info.get(self.name).get('ra')
        self.dec          = self.burst_info.get(self.name).get('dec')
        
        return
    
    
    def get_detectors(self, man_det: list) -> None:
        # Automatic detectors
        if self.detector_method == 'auto':
            self.detectors = self.burst_info.get(self.name).get('detectors')
            if len(self.detectors) == 0:
                raise ValueError('The burst has no automatic detectors. Please choose them manually!')
                
            
        # Pre-determined detectors
        elif self.detector_method == 'pre_fix':
            # Location of the 'pre_fix' file, a csv file with the detector names as 'n{i}' or b{i} where {i} is the number
            detector_file = open('{0}/{1}/pre_fix_detectors.txt'.format(self.data_path, self.number), 'r')
            detectors_tmp     = detector_file.read()
            self.detectors     = detectors_tmp.split(',')
            
        # Manual detectors
        elif self.detector_method == 'man':
            detector_file = open('{0}/{1}/pre_fix_detectors.txt'.format(self.data_path, self.number), 'w')
            
            for i in range(len(man_det)):  
                detector_file.write(man_det[i])

                if (i + 1) < len(man_det):
                    detector_file.write(',')
                    
            detector_file.close()
            self.detectors = man_det
            
        else:
            raise ValueError('{0} is not a valid method to choose detectors. Supported methods are: \'auto\', \'pre_fix\', and \'man\'.'.format(self.detector_method))
        
        self.num_detectors = len(self.detectors)
        log.info('Detectors: {0}\n'.format(self.detectors))
        
        return
    
    
    def download_fermi_data(self) -> None:
        
        files_to_download = []
        
        for detector in self.detectors:
            
            files_to_download.extend(self.get_detector_file_names(detector))
            
            
        # The url used to download data
        year = '20{0}'.format(self.number[0:2])
        url = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/{0}/bn{1}//current/'.format(year, self.number)
        
        # Gets the data from the url and finds all links with 'glg' in the name
        response = requests.get(url)
        data = bs4.BeautifulSoup(response.text, 'html.parser')
        all_filenames = data.find_all(href = re.compile('glg'))
            
        
        for file_to_download in files_to_download:
        
            for file in all_filenames:
                
                file_name = file['href']
                
                if file_name.startswith(file_to_download):
                    
                    file_name_path = '{0}/{1}/{2}'.format(self.data_path, self.number, file_name)
                    
                    file_exist = os.path.exists(file_name_path)
                
                    if file_exist:
                        log.info('File {0} already downloaded!\n'.format(file_name))
                    
                    else:
                        log.info('Downloads file {0}!\n'.format(file_name))
                        
                        file = requests.get(url + file_name)
                        
                        # If the download doesn't work properly
                        if file.status_code != 200:
                            raise Exception('File: {0} was not downloaded correctly'.format(file_name_path))
                        
                        # Saves the file in the data-folder
                        open(file_name_path, 'wb').write(file.content)
            
        return
    
    
    def get_detector_file_names(self, detector: str):
        
        file_var = [detector, self.number]
        
        if detector.lower() == 'lle':
            lle_file = 'gll_lle_bn{1}_v0'.format(*file_var)
            ft2_file = 'gll_pt_bn{1}_v0'.format(*file_var)
            rsp_file = 'gll_cspec_bn{1}_v0'.format(*file_var)
            
            file_names = [lle_file, ft2_file, rsp_file]
            
        else:
            tte_file_name = 'glg_tte_{0}_bn{1}_v0'.format(*file_var)
            rsp_file_name = 'glg_cspec_{0}_bn{1}_v0'.format(*file_var)
            
            file_names = [tte_file_name, rsp_file_name]
        
        return file_names
    
    
    def data_response_background(self, view_lc: bool, custom_start: float | None = None, custom_stop: float | None = None,
                                 poly_order_lle: int = 1, poly_order_gbm: int = 2, view_bg: bool = False, custom_lc: dict[str, bool | list | float] = {'custom' : False}) -> None:
        
        self.detector_data = {}
        
        if self.silence:
            silence_warnings()
        
        # Reads data files for given GRB, needs to be on your computer at data_path location
        self._read_data(poly_order_lle, poly_order_gbm)
        
        # Sets start and stop for data
        self._get_start_stop(custom_start, custom_stop)
        
        # If background intervals are automatic
        if self.bg_intervals == []:
            #Extracting information from burst_info
            background_interval   = self.burst_info.get('GRB' + self.number).get('background')
            self.bg_intervals  = [background_interval.get('pre'), background_interval.get('post')]
            
        # Plotting lightcurve, selection and background
        self._plot_lc(view_lc, view_bg, custom_lc)
        
        activate_warnings()
        activate_logs()
        
        return
    
    
    def _read_data(self, poly_order_lle: int, poly_order_gbm: int) -> None:
        
        for detector in self.detectors:
            if detector.lower() == 'lle':
                lle_file, ft2_file, lle_rsp = self.get_detector_file_versions(detector)
                
                
                self.detector_data[detector] = TimeSeriesBuilder.from_lat_lle(detector,
                                                        lle_file = lle_file,
                                                        ft2_file = ft2_file,
                                                        rsp_file = lle_rsp, poly_order = poly_order_lle)
            else:
                tte_file, rsp_file = self.get_detector_file_versions(detector)
                
                self.detector_data[detector] = TimeSeriesBuilder.from_gbm_tte(name = detector, tte_file = tte_file,
                                                                   rsp_file = rsp_file, poly_order = poly_order_gbm)
                
        return
    
    
    def get_detector_file_versions(self, detector: str):
        
        file_var = [self.data_path, self.number]
        
        if detector.lower() == 'lle':
            lle_file_name, ft2_file_name, rsp_file_name = self.get_detector_file_names(detector)
            
            lle_file = glob('{0}/{1}/{2}*.fit'.format(*file_var, lle_file_name))[0]
            ft2_file = glob('{0}/{1}/{2}*.fit'.format(*file_var, ft2_file_name))[0]
            rsp_file = glob('{0}/{1}/{2}*.rsp'.format(*file_var, rsp_file_name))[0]
            
            file_names = [lle_file, ft2_file, rsp_file]
            
        else:
            tte_file_name, rsp_file_name = self.get_detector_file_names(detector)
            
            tte_file = glob('{0}/{1}/{2}*.fit'.format(*file_var, tte_file_name))[0]
            
            try:
                rsp_file = glob('{0}/{1}/{2}*.rsp2'.format(*file_var, rsp_file_name))[0]
            except IndexError:
                rsp_file = glob('{0}/{1}/{2}*.rsp'.format(*file_var, rsp_file_name))[0]
        
            file_names = [tte_file, rsp_file]
        
        return file_names
    
    
    def _get_start_stop(self, custom_start: float, custom_stop: float) -> None:
        
        # If there is a custom start and stop
        if self.custom_start_stop:
            self.start    = custom_start
            self.stop     = custom_stop
            
        # If source_start and source_stop are automatic
        else:
            source_start_temp, source_stop_temp = self.burst_info.get('GRB' + self.number).get('source').get('fluence').split('0-')
            self.start = round(float(source_start_temp), 3)
            self.stop = round(float(source_stop_temp), 3)
        
        return
    
    
    def _plot_lc(self, view_lc: bool, view_bg: bool, custom_lc: dict[str, bool | list | float]) -> None:
        
        log.info('Fitting background:')
        activate_warnings()
        
        for detector in self.detectors:
            self.detector_data[detector].set_background_interval(*self.bg_intervals)
            self.detector_data[detector].set_active_time_interval('{0}-{1}'.format(self.start, self.stop))
            
            if custom_lc['custom']:
                fig = self.detector_data[detector].view_lightcurve(start = custom_lc['custom_times'][0], stop = custom_lc['custom_times'][1], dt = custom_lc['dt'])
            elif view_bg:
                fig = self.detector_data[detector].view_lightcurve(start = min(self.detector_data[detector]._time_series.bkg_intervals.starts) - 20, stop = max(self.detector_data[detector]._time_series.bkg_intervals.stops) + 20, dt = 0.1)
            else:
                fig = self.detector_data[detector].view_lightcurve(start = self.start - 2.0, stop = self.stop + 2.0, dt = 0.1)
            
            Plot.save_fig(fig, '{0}/figures/light_curves/light_curve_GRB{1}_detector_{2}.pdf'.format(self.results_path, self.number, detector))
            if view_lc:
                plt.show()
                plt.close()
            else:
                plt.close()
        
        return
    
    
    def create_time_bins(self, view_binned_lc, **kwargs):
        
        self._get_binning_detector_index(**kwargs)
        
        # Checks if bins have been made before, for the given parameters
        bins_path = '{0}/time_bins/time_bins_{1}_{2}_{3}_start_{4}_stop_{5}.txt'.format(self.results_path, self.binning_detector, *self.bin_method, self.start, self.stop)
        bins_exist = os.path.exists(bins_path)
        
        # If binning has already been done for the given parameters, load those bins. Else do the binning now.
        self._create_bins(bins_exist, bins_path, **kwargs)
        
        # Plots the lightcurve for the new time bins
        self._plot_time_bins(view_binned_lc)

        # Display the time bins
        log.info('Bins:')
        self.detector_data[self.detectors[0]].bins.display()
        
        # Finds the number of bins
        self.num_bins = len(self.detector_data[self.detectors[0]].bins)
        
        # Saves bins to file
        self._save_bins()
        
        return
    
    
    def _get_binning_detector_index(self, **kwargs):
        
        bin_by_detector = kwargs.pop('bin_by_detector', 'nai')
        
        if bin_by_detector == 'nai' or bin_by_detector == 'na' or bin_by_detector == 'n':
            if not any(detector.startswith('n') for detector in self.detectors):
                raise ValueError('No nai detector in list of chosen detectors.')
            self.binning_detector = [detector for detector in self.detectors if detector.startswith('n')][0]
            # self.binned_detector_index = [k for k, detector in enumerate(self.detectors) if detector.startswith('n')][0]
            
        elif bin_by_detector == 'bgo' or bin_by_detector == 'bg' or bin_by_detector == 'b':
            if not any(detector.startswith('b') for detector in self.detectors):
                raise ValueError('No bgo detector in list of chosen detectors.')
            self.binning_detector = [detector for detector in self.detectors if detector.startswith('b')][0]
            # self.binned_detector_index = [k for k, detector in enumerate(self.detectors) if detector.startswith('b')][0]
            
        elif bin_by_detector == 'lle':
            if not any(detector.startswith('lle') for detector in self.detectors):
                raise ValueError('No lle detector in list of chosen detectors.')
            self.binning_detector = [detector for detector in self.detectors if detector.startswith('lle')][0]
            # self.binned_detector_index = [k for k, detector in enumerate(self.detectors) if detector.startswith('lle')][0]
            
        else:
            raise ValueError('Invalid entry for detector to bin by. Use \"nai\", \"bgo\", or \"lle\". ')
            
        log.info('Binning by detector: {0}'.format(self.binning_detector))
            
        return
    
    
    def _create_bins(self, bins_exist: bool, bins_path: str, **kwargs) -> None:
        
        if bins_exist == True:
            log.info('Binning already made in the past. Reads file.\n')
            file = open(bins_path, 'r')
            i = 0
            for line in file.readlines():
                if i == 1:
                    custom_starts = line.split(',')
                if i == 3:
                    custom_stops = line.split(',')
                i += 1
            file.close
            
            self.detector_data[self.detectors[0]].create_time_bins(start = custom_starts, stop = custom_stops, method = 'custom', use_background = True)
            
        else:
            # Creates bins for the given method. It will only use the relevant variable of sigma/p0/dt
            log.info('Creating time bins:')
            if self.bin_method[0] == 'external':
                custom_bins = kwargs.pop('custom_bins')
                self.detector_data[self.binning_detector].create_time_bins(custom_bins[0], custom_bins[1], method = 'custom', use_background = True)
                
            else:
                self.detector_data[self.binning_detector].create_time_bins(self.start, self.stop, method = self.bin_method[0], sigma = self.bin_method[1], p0 = self.bin_method[1], dt = self.bin_method[1], use_background = True)
                    
                if self.binning_detector == 'lle':
                    
                    bad_bins = []
                    for i, w in enumerate(self.detector_data['lle'].bins.widths):
                        if w < 0.016:
                            bad_bins.append(i)
                    
                    edges = [self.detector_data['lle'].bins.starts[0]]
                    
                    for i, b in enumerate(self.detector_data['lle'].bins):
                        if i not in bad_bins:
                            edges.append(b.stop)
                    
                    starts = edges[:-1]
                    stops = edges[1:]
                    
                    self.detector_data['lle'].create_time_bins(starts, stops, method="custom")
                    
        if self.silence:
            silence_logs()
        for detector in self.detectors:
            self.detector_data[detector].read_bins(self.detector_data[self.binning_detector])
        activate_logs()
        
        return
    
    
    def _plot_time_bins(self, view_binned_lc: bool) -> None:
            
        for detector in self.detectors:
            fig = self.detector_data[detector].view_lightcurve(self.start, self.stop, use_binner = True)
            Plot.save_fig(fig, '{0}/figures/light_curves/light_curve_binned_GRB{1}_detector_{2}_{3}_{4}_start_{5}_stop_{6}.pdf'.format(self.results_path, self.number, detector, *self.bin_method, self.start, self.stop))
            if view_binned_lc:
                plt.show()
                plt.close()
            else:
                plt.close()
                
        return
    
    
    def _save_bins(self) -> None:
        
        file = open('{0}/time_bins/time_bins_{1}_{2}_{3}_start_{4}_stop_{5}_.txt'
                    .format(self.results_path, self.binning_detector, *self.bin_method, self.start, self.stop), 'w')
        file.write('Start times of every bin:')
        file.write('\n')
        
        start_times = ''
        stop_times = ''
        for i in range(self.num_bins):
            start_times += '{0}, '.format(round(self.detector_data[self.detectors[0]].bins[i].start_time, 3))
            stop_times  += '{0}, '.format(round(self.detector_data[self.detectors[0]].bins[i].stop_time,  3))
            
            if (i + 1) == self.num_bins:
                start_times = start_times[:-2]
                stop_times = stop_times[:-2]
                start_times += '\n'
            
        file.write(start_times)
        file.write('Stop times of every bin:')
        file.write('\n')
        file.write(stop_times)
        file.close()
        
        return
    
    
    def create_plugins(self, active_measurement_nai: list = ['8.-30.', '40.-850.'], active_measurement_bgo: list = ['250.-40000.'], active_measurement_lle: list = ['30000.-1000000.']) -> None:
        
        self.detector_plugins = {}
        
        if self.silence:
            silence_logs()
        
        for _, detector in enumerate(self.detectors):
            self.detector_plugins[detector] = self.detector_data[detector].to_spectrumlike(from_bins = True)
            
            for i in range(self.num_bins):
                if detector.startswith('n'):
                        self.detector_plugins[detector][i].set_active_measurements(*active_measurement_nai)
                                
                elif detector.startswith('b'):
                        self.detector_plugins[detector][i].set_active_measurements(*active_measurement_bgo)
                        
                elif detector.startswith('lle'):
                        self.detector_plugins[detector][i].set_active_measurements(*active_measurement_lle)

                else:
                    raise ValueError('There is a mysterious detector!')
        
        activate_logs()
        
        return
    
    
    def get_data(self):
        
        self.data = []
        for i in range(self.num_bins):
            self.data.append(DataList())
            for detector in self.detectors:
                self.data[i].insert(self.detector_plugins[detector][i])
                
        return


    def logging(self, load = False):
        
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank != 0:
            return
        
        if load:
            new_or_loaded = 'load'
        else:
            new_or_loaded = 'new'
        
        # Current date/time
        now = datetime.now()
        date = now.strftime('%Y-%m-%d')
        time = now.strftime('%H:%M:%S')
        
        # Define the path to the existing CSV file and check if it exists
        log_path = '{0}/log.txt'.format(self.results_path)
        log_exists = os.path.isfile(log_path)
        
        # Define the new row to be added
        headers = ['date', 'time', 'GRB', 'bin_method', 'bin_param', 'detec_method', 'detectors', 'start', 'stop', 'bg1', 'bg2', 'new_load']
        new_log_entry = [date, time, self.number, self.bin_method[0], self.bin_method[1], self.detector_method, '_'.join(self.detectors), self.start, self.stop, self.bg_intervals[0], self.bg_intervals[1], new_or_loaded]
        
        # Open the CSV file in append mode
        with open(log_path, 'a', newline = '') as log:
            # Create a csv.writer object
            writer = csv.writer(log)
            
            if not log_exists:
                writer.writerow(headers)
            
            # Write the new row at the bottom of the CSV file
            writer.writerow(new_log_entry)
        
        return
    
    
    def save(self):
        save_parameters = [self.burst_info, self.detectors, self.detector_data, self.detector_plugins, self.data]

        filename = self._get_filename(self.detectors, self.start, self.stop)
            
        misc.save_obj(save_parameters, filename)
        
        return
    
        
    def load(self, man_det, custom_start, custom_stop):
        
        filename = self._get_filename(man_det, custom_start, custom_stop)
            
        saved_parameters = misc.load_obj(filename)
        
        self.burst_info = saved_parameters[0]
        self.get_info(load = True)
        self.detectors = saved_parameters[1]
        self.num_detectors = len(self.detectors)
        self.detector_data = saved_parameters[2]
        self.start = self.detector_data[self.detectors[0]].tstart
        self.stop = self.detector_data[self.detectors[0]].tstop
        self.bg_intervals = ['{0}-{1}'.format(self.detector_data[self.detectors[0]]._time_series.bkg_intervals[i].start, self.detector_data[self.detectors[0]]._time_series.bkg_intervals[i].stop) for i, _ in enumerate(self.detector_data[self.detectors[0]]._time_series.bkg_intervals)]
        self.num_bins = len(self.detector_data[self.detectors[0]].bins)
        self.detector_plugins = saved_parameters[3]
        self.data = saved_parameters[4]
        
        self.logging(load = True)
        
        return
    
    
    def _get_filename(self, det, start, stop):
        
        if self.detector_method == 'man':
            if self.custom_start_stop:
                filename = '{0}/setups/setup_for_{1}_{2}_{3}_{4}_start_{5}_stop_{6}'.format(self.results_path, self.number, *self.bin_method, '_'.join(det), start, stop)
            else:
                filename = '{0}/setups/setup_for_{1}_{2}_{3}_{4}_start_{5}_stop_{6}'.format(self.results_path, self.number, *self.bin_method, '_'.join(det), 'auto', 'start_stop')
        elif self.detector_method == 'auto': 
            if self.custom_start_stop:
                filename = '{0}/setups/setup_for_{1}_{2}_{3}_{4}_start_{5}_stop_{6}'.format(self.results_path, self.number, *self.bin_method, self.detector_method, start, stop)
            else:
                filename = '{0}/setups/setup_for_{1}_{2}_{3}_{4}_start_{5}_stop_{6}'.format(self.results_path, self.number, *self.bin_method, self.detector_method, 'auto', 'start_stop')
                
        else:
            raise Exception('Chosen detector_method is not supported.')
                
        return filename

    
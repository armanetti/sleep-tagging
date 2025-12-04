from montage import create_montage
import mne
from mne.preprocessing import ICA
import numpy as np
import pandas as pd
import pyprep

def make_preprocessing(edf_fname, mont_fname, csv_fname):
    raw = mne.io.read_raw_edf(edf_fname, eog=['EOG1', 'EOG2'], 
                              misc=['RLEG-', 'RLEG+', 'LLEG-', 'LLEG+'], 
                              preload=False)
    
    resample_rate = 128
    
    csv = pd.read_csv(csv_fname, sep='\t', header=None)
    idxs = csv.loc[:, csv.columns[0]].values
    scoring_time = csv.loc[:, csv.columns[1]].values[idxs!='L']
    tmin = scoring_time[0]
    tmax = scoring_time[-1] + 30
    
    raw.resample(resample_rate, n_jobs=8)
    
    raw.crop(tmin=tmin, tmax=tmax)
        
    mapping = {'ChEMG1': 'emg',
               'ChEMG2': 'emg',
               'ECG1': 'ecg',
               'ECG2': 'ecg'}
    
    raw.set_channel_types(mapping)
    raw.rename_channels({'C1-': 'C1'})
    
    montage = create_montage(mont_fname)
    raw.set_montage(montage, match_case=False)
    
    raw = raw.filter(l_freq=.3, h_freq=45., picks='eeg', n_jobs='cuda')
    raw = raw.filter(l_freq=.5, h_freq=40., picks='ecg', n_jobs='cuda')
    raw = raw.filter(l_freq=10., h_freq=None, picks='emg', n_jobs='cuda')
    raw = raw.filter(l_freq=.1, h_freq=30., picks='eog', n_jobs='cuda')
    
    raw = raw.set_eeg_reference(ref_channels='average')
    
    nc = pyprep.NoisyChannels(raw=raw.copy().pick_types(eeg=True).
                                  resample(100.), 
                                  do_detrend=True, random_state=23, 
                                  matlab_strict=False)
    nc.find_all_bads()
    nc.find_bad_by_nan_flat()
    bads = nc.get_bads(as_dict=True)
    
    raw.info['bads'] = bads['bad_all']
    
    ica = ICA(n_components=30, noise_cov=None, random_state=23, 
              method='fastica', max_iter=1500, allow_ref_meg=False)
    ica.fit(inst=raw, decim=1, reject=None)
    
    bad_ecg1, ecg_scores1 = ica.find_bads_ecg(raw, ch_name='ECG1', method='ctps', measure='correlation', threshold=.1)
    bad_ecg2, ecg_scores2 = ica.find_bads_ecg(raw, ch_name='ECG2', method='ctps', measure='correlation', threshold=.2)
    bad_emg, emg_scores = ica.find_bads_muscle(raw, sphere='auto',
                                               threshold=0.4)
    
    bads_ica = list(np.unique(bad_ecg1 + bad_ecg2 + bad_emg))
    ica.exclude = bads_ica
    
    prep_raw = ica.apply(raw, exclude=bads_ica)
    
    prep_raw.set_eeg_reference(ref_channels='average')
    
    prep_raw.save('/home/jerry/Documenti/Research/BrainHack/2025/project/EPCTL06/EPCTL06-prep.fif', overwrite=True)
    
    return prep_raw
    
        
if __name__ == '__main__':
    edf_fname = '/home/jerry/Documenti/Research/BrainHack/2025/project/EPCTL06/EPCTL06.edf'
    montage_fname = '/home/jerry/Documenti/Research/BrainHack/2025/project/average.pos'
    csv_fname = '/home/jerry/Documenti/Research/BrainHack/2025/project/EPCTL06/EPCTL06.txt'
    raw = make_preprocessing(edf_fname, montage_fname, csv_fname)
    print(raw)
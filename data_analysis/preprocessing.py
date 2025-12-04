from montage import create_montage
import mne
from mne.preprocessing import ICA
import numpy as np

def make_preprocessing(edf_fname, mont_fname):
    raw = mne.io.read_raw_edf(edf_fname, eog=['EOG1', 'EOG2'], 
                              misc=['RLEG-', 'RLEG+', 'LLEG-', 'LLEG+'], 
                              preload=False)
    
    raw.resample(128, n_jobs=8)
    
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
    
    ica = ICA(n_components=.95, noise_cov=None, random_state=23, 
              method='fastica', max_iter=1500, allow_ref_meg=False)
    ica.fit(inst=raw, decim=1, reject=None)
    
    bad_ecg1, ecg_scores1 = ica.find_bads_ecg(raw, ch_name='ECG1')
    bad_ecg2, ecg_scores2 = ica.find_bads_ecg(raw, ch_name='ECG2')
    bad_emg, emg_scores = ica.find_bads_muscle(raw, sphere='auto',
                                               threshold=0.1)
    
    bads_ica = list(np.unique(bad_ecg1 + bad_ecg2 + bad_emg))
    ica.exclude = bads_ica
    
    prep_raw = ica.apply(raw, exclude=bads_ica)
    
    prep_raw.set_eeg_reference(ref_channels='average')
    
    prep_raw.save('/home/jerry/Documenti/Research/BrainHack/2025/project/EPCTL06/EPCTL06-prep.fif', overwrite=True)
    
    return prep_raw
    
    
    
if __name__ == '__main__':
    edf_fname = '/home/jerry/Documenti/Research/BrainHack/2025/project/EPCTL06/EPCTL06.edf'
    raw = make_preprocessing(edf_fname, 
                            '/home/jerry/Documenti/Research/BrainHack/2025/project/average.pos')
    print(raw)
import os
import os.path as op
import pandas as pd
import mne


def create_montage(avg_montage):
    csv = pd.read_csv(avg_montage, sep='\t', header=None, index_col=1)
    csv.columns = ['n', 'X', 'Y', 'Z']
    ch_names = csv.index.tolist()
    ch_positions = csv[['X', 'Y', 'Z']].values
    channels = dict(zip(ch_names, ch_positions))
    montage = mne.channels.make_dig_montage(ch_pos=channels, 
                                            coord_frame='head')
    return montage

if __name__ == '__main__':
    avg_montage = '/home/jerry/Documenti/Research/BrainHack/2025/project/average.pos'
    montage = create_montage(avg_montage)
    # montage.save(op.join('data', 'custom_montage.fif'))
    print(montage)
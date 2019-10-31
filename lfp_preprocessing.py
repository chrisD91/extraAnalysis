import sys, os
sys.path.append('/home/yzerlaut/work/data_analysis/IO')
import hdf5
import numpy as np

from elphy_reader import ElphyFile, realign_episode_data_over_time
from wavelet import my_cwt

def compute_band_envelope(data, dt, band=[30., 80.]):
    """

    """
    freqs = np.linspace(band[0], band[1], 10)
    return np.max(np.abs(my_cwt(data, freqs, dt)), axis=0)


if __name__ == "__main__":

    if len(sys.argv)<2:
        print('need to give a datafile as an argument, e.g.: "python lfp_preprocessing.py /media/yzerlaut/Elements/4319_CXRIGHT/4319_CXRIGHT_NBR1.DAT" ')
    else:
        filename = sys.argv[-1]
        ef = ElphyFile(filename, read_data=False, read_spikes=False)

        new_dt = 2e-3
        new_dt_for_mua = 1e-4
        new_dt_for_gamma = 1e-3

        # let's do it once for the first channel
        Nchannel, ichannel = 1, 32
        # LFP (subsampled Extracellular signal)
        subsmpl_data = realign_episode_data_over_time(ef, ichannel, subsampling=new_dt)
        lfp = subsmpl_data
        # gamma power
        subsmpl_data=realign_episode_data_over_time(ef,ichannel,subsampling=new_dt_for_gamma)
        gamma_power = compute_band_envelope(subsmpl_data, new_dt_for_gamma, band=[30., 80.])[::int(new_dt/new_dt_for_gamma)]
        # MUA
        subsmpl_data = realign_episode_data_over_time(ef,ichannel,subsampling=new_dt_for_mua)
        MUA = compute_band_envelope(subsmpl_data, new_dt_for_mua, band=[300., 3e3])[::int(new_dt/new_dt_for_mua)]

        
        # we sample 20 of the 64 channels
        for ichannel in np.random.randint(64, size=20):
            # LFP (subsampled Extracellular signal)
            subsmpl_data = realign_episode_data_over_time(ef, ichannel, subsampling=new_dt)
            lfp = subsmpl_data
            # gamma power
            subsmpl_data=realign_episode_data_over_time(ef,ichannel,subsampling=new_dt_for_gamma)
            gamma_power = compute_band_envelope(subsmpl_data, new_dt_for_gamma, band=[30., 80.])[::int(new_dt/new_dt_for_gamma)]
            # MUA
            subsmpl_data = realign_episode_data_over_time(ef,ichannel,subsampling=new_dt_for_mua)
            MUA = compute_band_envelope(subsmpl_data, new_dt_for_mua, band=[300., 3e3])[::int(new_dt/new_dt_for_mua)]
            Nchannel += 1

        
        np.savez(filename.replace('.DAT', os.path.sep+'preprocessed_extra.npz'),
                 **{'lfp':lfp/Nchannel, 'gamma_power':gamma_power/Nchannel, 'MUA':MUA/Nchannel, 'dt':new_dt})
        


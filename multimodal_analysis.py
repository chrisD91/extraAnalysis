import sys, os
import numpy as np
import hdf5
import matplotlib.pylab as plt

folder = sys.argv[1]


extra_signals = np.load(folder+os.path.sep+'preprocessed_extra.npz')

dt = 1./3e4
# basis = hdf5.load_dict_from_hdf5(folder+os.path.sep+folder+'.basis.hdf5')
# result = hdf5.load_dict_from_hdf5(folder+os.path.sep+folder+'.result.hdf5')
result_merged = hdf5.load_dict_from_hdf5(folder+os.path.sep+folder+'.result-merged.hdf5')
# clusters = hdf5.load_dict_from_hdf5(folder+os.path.sep+folder+'.clusters.hdf5')
# templates = hdf5.load_dict_from_hdf5(folder+os.path.sep+folder+'.templates.hdf5')
# clusters_merged = hdf5.load_dict_from_hdf5(folder+os.path.sep+folder+'.clusters-merged.hdf5')
# templates_merged = hdf5.load_dict_from_hdf5(folder+os.path.sep+folder+'.templates-merged.hdf5')
# overlap = hdf5.load_dict_from_hdf5(folder+os.path.sep+folder+'.overlap.hdf5')


def show_raster_plot(result=result_merged):
    for i, nrn in enumerate(result['spiketimes'].keys()):
        plt.plot(result['spiketimes'][nrn]*dt, i*np.ones(len(result['spiketimes'][nrn])), 'o', ms=2)
    plt.show()


def show_extra_multimodal(result,
                          extra_signals,
                          zoom=[0, 5]):

    fig, AX = plt.subplots(4, 1, figsize=(14,8))
    plt.subplots_adjust(top=.99)
    sbsmpl_t = extra_signals['dt']*np.arange(len(extra_signals['lfp']))
    cond = (sbsmpl_t>zoom[0]) & (sbsmpl_t<zoom[1])
    AX[0].plot(sbsmpl_t[cond], extra_signals['lfp'][cond], 'k-')
    AX[2].plot(sbsmpl_t[cond], extra_signals['gamma_power'][cond], 'k-')
    sbsmpl_t = extra_signals['dt']*np.arange(len(extra_signals['MUA']))
    cond = (sbsmpl_t>zoom[0]) & (sbsmpl_t<zoom[1])
    AX[1].plot(sbsmpl_t[cond], extra_signals['MUA'][cond], 'k-')

    for i, nrn in enumerate(result['spiketimes'].keys()):
        spikes = result['spiketimes'][nrn]*dt
        spikes_in_window = spikes[(spikes>zoom[0]) & (spikes<zoom[1])]
        AX[3].plot(spikes_in_window, i*np.ones(len(spikes_in_window)), 'ko', ms=1)
        # AX[3].plot(spikes, i*np.ones(len(spikes)), 'ko', ms=1)
        AX[3].plot(zoom, [i,i], 'wo', ms=1)
    for ax, label in zip(AX, ['LFP', 'MUA', 'Gamma-Pow', 'Neuron ID']):
        ax.set_ylabel(label)
    AX[3].set_xlabel('time (s)')
    plt.show()
    
def show_firing_rate_plot(result=result_merged):
    
    plt.plot(result['spiketimes'][nrn]*dt, i*np.ones(len(result['spiketimes'][nrn])), 'o')
    plt.show()
    

show_extra_multimodal(result_merged, extra_signals,
                      zoom=[float(sys.argv[2]), float(sys.argv[3])])
# show_raster_plot(result=result_merged)
# print(basis['waveforms'].shape, result_merged['amplitudes']['temp_0'].shape, basis['waveforms']['temp_0'].shape)

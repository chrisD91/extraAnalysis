import sys
sys.path.append('/home/yzerlaut/work/data_analysis/IO')
import hdf5
import numpy as np

folder = '4319_CXRIGHT_NBR1'
dt = 1./3e4
basis = hdf5.load_dict_from_hdf5(folder+'.basis.hdf5')
result = hdf5.load_dict_from_hdf5(folder+'.result.hdf5')
result_merged = hdf5.load_dict_from_hdf5(folder+'.result-merged.hdf5')
clusters = hdf5.load_dict_from_hdf5(folder+'.clusters.hdf5')
templates = hdf5.load_dict_from_hdf5(folder+'.templates.hdf5')
clusters_merged = hdf5.load_dict_from_hdf5(folder+'.clusters-merged.hdf5')
templates_merged = hdf5.load_dict_from_hdf5(folder+'.templates-merged.hdf5')
overlap = hdf5.load_dict_from_hdf5(folder+'.overlap.hdf5')

import matplotlib.pylab as plt

def show_raster_plot(result=result):
    for i, nrn in enumerate(result['spiketimes'].keys()):
        plt.plot(result['spiketimes'][nrn]*dt, i*np.ones(len(result['spiketimes'][nrn])), 'o')
    plt.show()

def show_firing_rate_plot(result=result):
    
    plt.plot(result['spiketimes'][nrn]*dt, i*np.ones(len(result['spiketimes'][nrn])), 'o')
    plt.show()
    
def show_templates(templates=templates):
    print(templates.keys())
    # for i, nrn in enumerate(result['spiketimes'].keys()):
    #     print(result['spiketimes'][nrn])
    #     plt.plot(result['spiketimes'][nrn], i*np.ones(len(result['spiketimes'][nrn])), 'o')
    
    
show_raster_plot(result=result_merged)
show_templates(templates=templates)
# print(basis['waveforms'].shape, result_merged['amplitudes']['temp_0'].shape, basis['waveforms']['temp_0'].shape)

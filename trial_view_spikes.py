import sys, os
import numpy as np
import hdf5
import matplotlib.pylab as plt
from matplotlib import colorbar

folder = sys.argv[1] # sets the datafile (by point to a folder containing all the analysis)

dt = 1./3e4
result_merged = hdf5.load_dict_from_hdf5(folder+os.path.sep+folder+'.result-merged.hdf5')


def show_raster_plot(result=result_merged,
                     stim_duration = 10.,
                     pre_stim_duration = 2.,
                     post_stim_duration = 2.,
                     stim_periodicity = 25):

    fig, AX = plt.subplots(5, 5, figsize=(10, 10))
    cycle_duration = stim_duration+pre_stim_duration+post_stim_duration
    tmax = np.max(np.concatenate([result['spiketimes'][key] for key in result['spiketimes']])*dt)+cycle_duration # time of last spike + cycle duration for security
    print("with a stimulus periodicity of %i, %i repetitions were found" % (stim_periodicity, tmax/cycle_duration/stim_periodicity))
    for stim in np.arange(stim_periodicity):
        for i, nrn in enumerate(result['spiketimes'].keys()): # loop over neurons
            spiketimes = result['spiketimes'][nrn]*dt
            start = stim*cycle_duration # time of first trial of that stimulus (of id: "stim")
            while start<tmax:
                cond = (spiketimes>start) & (spiketimes<start+cycle_duration)
                AX.flatten()[stim].plot(spiketimes[cond]-start-pre_stim_duration, i+.2*np.random.randn(len(spiketimes[cond])), 'o',
                         ms=0.2, color=plt.cm.viridis(start/tmax))
                start += cycle_duration*stim_periodicity
                AX.flatten()[stim].set_title('stim %i' % (stim+1))

        plt.yticks(range(i), [])
    AX[0][0].set_ylabel('neuron ID')
    AX[4][4].set_xlabel('time (s)')


    fig, ax= plt.subplots(1, figsize=(5,2))
    plt.subplots_adjust(bottom=.4)
    colorbar.ColorbarBase(ax=ax, cmap=plt.cm.viridis,
                          orientation="horizontal")
    ax.axis('off')
    ax.annotate('Trial ID', (0.5, 0.2), xycoords='axes fraction', ha='center', va='center')
    plt.show()

show_raster_plot()



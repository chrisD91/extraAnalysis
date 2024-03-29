import sys, os
import numpy as np
import hdf5
import matplotlib.pylab as plt

folder = sys.argv[1] # sets the datafile (by point to a folder containing all the analysis)

extra_signals = np.load(folder+os.path.sep+'preprocessed_extra.npz')

def show_extra_plot(signal_key='MUA', # stimulus ID
                    stim_duration = 10.,
                    pre_stim_duration = 2.,
                    post_stim_duration = 2.,
                    stim_periodicity = 25):

    cycle_duration = (stim_duration+pre_stim_duration+post_stim_duration)

    # raw data responses
    fig_raw, AX = plt.subplots(25, 1, figsize=(15,14))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.001, hspace=0.001)

    # cross-correlations
    fig_cc, ax = plt.subplots(1, figsize=(10,5))
    plt.subplots_adjust(bottom=0.4)

    sbsmpl_t = extra_signals['dt']*np.arange(len(extra_signals['lfp']))
    for iax, istim in enumerate(np.arange(stim_periodicity)):
        start = istim*cycle_duration # time of first trial of that stimulus (of id: "stim")
        RESP = []
        while start<=sbsmpl_t[-1]:
            cond = (sbsmpl_t>start) & (sbsmpl_t<start+cycle_duration)
            RESP.append(extra_signals[signal_key][cond])
            start += cycle_duration*stim_periodicity
        # showing now mean+s.e.m
        mean, std = np.mean(RESP, axis=0), np.std(RESP, axis=0)
        AX[iax].plot(sbsmpl_t[cond]-sbsmpl_t[cond][0], mean, 'k-', lw=2) # showing raw trace
        AX[iax].fill_between(sbsmpl_t[cond]-sbsmpl_t[cond][0], mean-std, mean+std, color='k', alpha=.2) # showing raw trace
        # calculating mean correlation
        CC = []
        # stupid imlementation:
        for i, x1 in enumerate(RESP):
            for j, x2 in enumerate([RESP[ii] for ii in range(i+1, len(RESP))]):
                CC.append(float(np.corrcoef(x1, x2)[0,1]))
        ax.bar([istim], np.mean(CC), yerr=np.std(CC), color='lightgray', ecolor='k', lw=1)
        plt.xticks(range(stim_periodicity), ['stim %i' % (istim+1) for istim in range(stim_periodicity)], rotation=70)
        AX[iax].annotate('stim %i' % (istim+1), (0.,0.5), xycoords='axes fraction')
        ax.set_ylabel('trial-to-trial correlation\n(mean $\pm$ s.e.m over %ix%i pairs)' % (len(RESP), len(RESP)-1))
        ax.set_title('Signal: '+signal_key)
    plt.show()

    
show_extra_plot(sys.argv[-1])



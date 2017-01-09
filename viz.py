import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

class viz:
    def block_reconstruction_error(block_errors, test_errors, net_names):
        '''
            BLOCK_RECONSTRUCTION_ERROR visualizes the reconstruction error
             over training blocks

            INPUTS:
                block_errors = [list] list of training reconstruction error
                                dictionaries for the different models
                test_errors  = [list] list of test reconstruction error
                                dictionaries for the different models
                net_names    = [list] list of strings for labeling the
                                plot series
        '''

        nblocks = len(block_errors[0])
        nnets = len(block_errors)
        block_stats = np.zeros([nblocks+1, 2, nnets])

        fig, ax = plt.subplots(1, 1)

        for i in range(nnets):

            for j in range(nblocks):
                mean, var, std = stats.mvsdist(block_errors[i][j]['m'])
                block_stats[j,0,i] = mean.mean()
                block_stats[j,1,i] = np.abs(mean.mean() - mean.interval(0.95)[0])

            mean, var, std = stats.mvsdist(test_errors[i]['m'])
            block_stats[nblocks,0,i] = mean.mean()
            block_stats[nblocks,1,i] = np.abs(mean.mean() - mean.interval(0.95)[0])

            ax.errorbar(np.arange(nblocks+1), block_stats[:,0,i], yerr=block_stats[:,1,i], label=net_names[i])

        ax.set_xlim([-1, nblocks+1])
        ax.axvline(nblocks-0.5, linestyle='--', color='k')
        #ax.set_ylim([0, 1])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_title('Reconstruction Accuracy over Blocks')
        plt.show()

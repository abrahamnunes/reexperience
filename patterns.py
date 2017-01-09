import numpy as np
import matplotlib.pyplot as plt

def generate_patterns(nvisible, npatterns, p):
    pat = np.random.binomial(1, p, size=[nvisible, npatterns])
    return pat

def generate_nested_patterns(nvisible, nclasses, nsubclasses, npatterns, p):
    '''
        GENERATE_NESTED_PATTERNS creates pattern vectors based on templates
         in a hierarchical fashion

        INPUTS:
            nvisible    = [int] number of visible units in the RBM
            nclasses    = [int] number of top-level classes
            nsubclasses = [int] number of subclasses per top-level class
            npatterns   = [int] number of patterns per subclass
            p           = [scalar] proportion of "active" units per pattern.

        OUTPUT:
            pat = [dict] dictionary of class and subclass templates, as well
                    as pattern vectors
    '''

    # Preallocate
    pat = {
        'class'      : np.zeros([nvisible, nclasses]),
        'subclass'   : np.zeros([nvisible, nclasses*nsubclasses]),
        'pattern'   : np.zeros([nvisible, nclasses*nsubclasses*npatterns]),
        'pattern_id' : np.zeros([nclasses*nsubclasses*npatterns, 3])
    }

    # Create class templates, subclass templates, and patterns
    subclass_count  = 0
    pattern_count   = 0
    non_overlapping = False
    while non_overlapping is False:
        for i in range(nclasses):
            pat['class'][:, i] = np.random.binomial(1, p, size=nvisible)
            for j in range(nsubclasses):
                prob = np.zeros(nvisible)
                prob[pat['class'][:,i] == 1] = 0.8
                prob[pat['class'][:,i] == 0] = 0.02
                pat['subclass'][:, subclass_count] = np.random.binomial(1, prob)

                for k in range(npatterns):
                    prob = np.zeros(nvisible)
                    prob[pat['subclass'][:,subclass_count] == 1] = 0.8
                    prob[pat['subclass'][:,subclass_count] == 0] = 0.02
                    pat['pattern'][:, pattern_count] = np.random.binomial(1, prob)

                    pat['pattern_id'][pattern_count,:] = [i, j, k]

                    pattern_count += 1

                subclass_count += 1

        overlap = []
        for i in range(nclasses-1):
            for j in range(i+1, nclasses):
                overlap.append(np.sum(np.abs(pat['class'][:,i] - pat['class'][:,i])))

        non_overlapping = np.all(overlap == np.max(overlap))

    return pat

def split_training_test(pat, ptest):
    '''
        SPLIT_TRAINING_TEST splits the patterns into training and test sets

        INPUTS:
            pat   = [dict] dictionary of class and subclass templates, as
                     well as pattern vectors
            ptest = [scalar] proportion of the data allocated to test set

        OUTPUTS:
            pat = [dict] dictionary of classes, subclasses, and patterns,
                now split further into training and test elements at
                the pattern level
    '''

    npatterns = np.shape(pat['pattern'])[1]
    ntrain = int(np.floor(npatterns*(1-ptest)))
    ind = np.arange(npatterns)
    np.random.shuffle(ind)

    training_patterns = pat['pattern'][:,0:ntrain]
    test_patterns = pat['pattern'][:,ntrain:]

    training_pattern_id = pat['pattern_id'][0:ntrain,:]
    test_pattern_id = pat['pattern_id'][ntrain:,:]

    pat2 = {
        'class'      : pat['class'],
        'subclass'   : pat['subclass'],
        'pattern'    : {
            'train' : training_patterns,
            'test'  : test_patterns
        },
        'pattern_id' : {
            'train' : training_pattern_id,
            'test'  : test_pattern_id
        }
    }

    return pat2

def visualize_patterns(pat):
    '''
        VISUALIZE_PATTERNS shows an image of the patterns

        INPUTS:
            pat = [dict] dictionary of classes and patterns
    '''

    if 'test' in pat['pattern']:
        fig, ax = subplots(1, 2)
        ax[0].imshow(pat['pattern']['train'], cmap='viridis')
        ax[0].title('Average activation (Training): ' + str(np.round(np.mean(np.mean(pat['pattern']['train'])), 2)))
        ax[0].xlabel('Pattern')
        ax[0].ylabel('Unit')

        ax[1].imshow(pat['pattern']['test'], cmap='viridis')
        ax[1].title('Average activation (Test): ' + str(np.round(np.mean(np.mean(pat['pattern']['test'])), 2)))
        ax[1].xlabel('Pattern')
        ax[1].ylabel('Unit')
        plt.show()
    else:
        plt.figure()
        plt.imshow(pat['pattern'], cmap='viridis')
        plt.title('Average activation: ' + str(np.mean(np.mean(pat['pattern']))))
        plt.xlabel('Pattern')
        plt.ylabel('Unit')
        plt.show()

def visualize_clusters(pat):
    '''
        TEST_PATTERN_CORRELATION examines the correlation between patterns and their underlying classes by plotting them as a correlation matrix
    '''

    import scipy.cluster.hierarchy as sch

    if 'test' in pat['pattern']:
        nunits = np.shape(pat['pattern']['train'])[0]
        ntrain = np.shape(pat['pattern']['train'])[1]
        ntest  = np.shape(pat['pattern']['test'])[1]

        # Compute distance matrix for training patterns
        Dtrain = np.zeros([ntrain, ntrain])
        for i in range(ntrain):
            for j in range(ntrain):
                Dtrain[i, j] = np.sum(np.abs(pat['pattern']['train'][:,i] - pat['pattern']['train'][:,j])/nunits)

        # Compute distance matrix for test patterns
        Dtest = np.zeros([ntest, ntest])
        for i in range(ntest):
            for j in range(ntest):
                Dtest[i, j] = np.sum(np.abs(pat['pattern']['test'][:,i] - pat['pattern']['test'][:,j])/nunits)

        # Plot dendrograms (Training)
        fig = plt.figure()
        axdendro = fig.add_axes([0.09, 0.1, 0.2, 0.8])
        Y = sch.linkage(Dtrain, method='weighted')
        Z = sch.dendrogram(Y, orientation='left')
        axdendro.set_xticks([])
        axdendro.set_yticks([])

        # Plot distance matrix (Training)
        axmatrix=fig.add_axes([0.3, 0.1, 0.6, 0.8])
        index=Z['leaves']
        Dtrain = Dtrain[index,:]
        Dtrain = Dtrain[:,index]
        im = axmatrix.matshow(Dtrain, aspect='auto', origin='lower', cmap='viridis')
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])
        axmatrix.set_title('Training Data')

        fig.show()

        # Plot dendrograms (Test)
        fig = plt.figure()
        axdendro = fig.add_axes([0.09, 0.1, 0.2, 0.8])
        Y = sch.linkage(Dtest, method='weighted')
        Z = sch.dendrogram(Y, orientation='left')
        axdendro.set_xticks([])
        axdendro.set_yticks([])

        # Plot distance matrix (Test)
        axmatrix=fig.add_axes([0.3, 0.1, 0.6, 0.8])
        index=Z['leaves']
        Dtest = Dtest[index,:]
        Dtest = Dtest[:,index]
        im = axmatrix.matshow(Dtest, aspect='auto', origin='lower', cmap='viridis')
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])
        axmatrix.set_title('Test Data')

        fig.show()

    else:
        nunits    = np.shape(pat['pattern'])[0]
        npatterns = np.shape(pat['pattern'])[1]

        # Compute distance matrix for training patterns
        D = np.zeros([npatterns, npatterns])
        for i in range(npatterns):
            for j in range(npatterns):
                D[i, j] = np.sum(np.abs(pat['pattern'][:,i] - pat['pattern'][:,j])/nunits)

        # Plot dendrograms (Training)
        fig = plt.figure()
        axdendro = fig.add_axes([0.09, 0.1, 0.2, 0.8])
        Y = sch.linkage(D, method='weighted')
        Z = sch.dendrogram(Y, orientation='left')
        axdendro.set_xticks([])
        axdendro.set_yticks([])

        # Plot distance matrix (Training)
        axmatrix=fig.add_axes([0.3, 0.1, 0.6, 0.8])
        index=Z['leaves']
        D = D[index,:]
        D = D[:,index]
        im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap='viridis')
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])
        axmatrix.set_title('All Patterns')

        fig.show()

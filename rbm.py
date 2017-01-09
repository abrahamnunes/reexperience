import numpy as np
import numpy.matlib as matlib

class rbm:
    def __init__(self, name, neurogenesis, nvisible, nhidden, lr, tgt_sparsity, sparsity_cost, weight_decay, niter):
        '''
            Initialize the network

            INPUTS:
                name          = [string] the name of the network
                neurogenesis  = [bool] whether the network has neurogenesis
                nvisible      = [int] number of visible units
                nhidden       = [int] number of hidden units
                lr            = [1 X 2] learning rate range
                tgt_sparsity  = [scalar] target sparsity of hidden units
                sparsity_cost = [1 X 2] sparsity cost range
                weight_decay  = [scalar] weight decay parameter
                npatterns     = [int] number of patterns in inputs
        '''

        # Instantiate network weights (w), biases (bv, bh for visible and hidden, respectively), learning rate (lr)
        self.nvisible      = nvisible
        self.nhidden       = nhidden
        self.neurogenesis  = neurogenesis
        self.w                  = np.random.normal(0, 0.1, size=[nhidden, nvisible])
        self.w_mask             = np.ones([nhidden, nvisible])
        self.w                  = self.w * self.w_mask
        self.weight_decay       = weight_decay

        self.bv = np.zeros(nvisible)
        self.bh = np.zeros(nhidden)

        self.lr_range = lr
        self.lr       = np.zeros(nhidden) + lr[1]

        self.tgt_sparsity        = tgt_sparsity
        self.sparsity_cost_range = sparsity_cost
        self.sparsity_cost       = np.zeros(nhidden) + sparsity_cost[0]

        self.age            = np.zeros(nhidden)
        self.age[1:nhidden] = 1

        # Instantiate simulation parameters and data storage elements
        self.name  = name
        self.niter = niter
        self.error = {}

    def activation(self, x):
        '''
            ACTIVATION simulates the sigmoid activation function

            INPUTS:
                x   = the neuronal inputs + biases

            OUTPUTS:
                y = the resulting neuron activations
        '''
        y = (1/(1 + np.exp(-x)) > np.random.uniform(0, 1, size=np.shape(x))) * 1

        return y

    def cd_up(self, v):
        '''
            CD_UP computes the up phase of contrastive divergence

            INPUTS:
                v = [nvisible X 1] vector of visible unit activations

            OUTPUTS:
                h = [nhidden X 1] vector of hidden unit activations
        '''

        h = self.activation(self.bh + np.dot(self.w, v))

        return h

    def cd_down(self, h):
        '''
            CD_DOWN computes the down phase of contrastive divergence

            INPUTS:
                h = [nhidden X 1] vector of hidden unit activations

            OUTPUTS:
                v_recon = [nvisible X 1] vector of reconstructed visible
                            unit activations
                h_recon = [nhidden X 1] vector of hidden unit activations
                            driven by reconstructed visible units
        '''

        v_recon = self.activation(self.bv + np.dot(self.w.T, h))
        h_recon = self.activation(self.bh + np.dot(self.w, v_recon))

        return v_recon, h_recon

    def growth(self, t, nblocks):
        '''
            GROWTH models neurogenesis between training blocks

            INPUTS:

            OUTPUTS:
                self = rbm object with updated weight mask, learning rates,
                        and sparsity costs

        '''

        '''
        start_idx = int(self.nhidden+(t*nnewneurons))
        end_idx   = int(self.nhidden+((t+1)*nnewneurons))

        self.w[start_idx:end_idx, :] = np.random.normal(0, 0.1, size=np.shape(self.w[start_idx:end_idx, :]))
        self.w_mask[start_idx:end_idx, :] = np.ones(np.shape(self.w_mask[start_idx:end_idx, :]))
        '''
        alpha = 0.2
        beta  = 0.65
        gamma   = 0.15

        strength        = np.mean(np.abs(self.w), axis=1)
        differentiation = np.std(np.abs(self.w), axis=1)
        Z = (alpha*strength + beta*differentiation + gamma*self.age)/(alpha+beta+gamma)

        #Increment age
        self.age += 1

        #Reset ages of neurons with Z in lowest 5%
        self.age[Z < np.percentile(Z, 5)] = 0

        #Update neuron learning rates
        lr_change = (self.lr_range[1] - self.lr_range[0])/nblocks
        self.lr = self.lr - (self.age > 0)*lr_change
        self.lr[Z < np.percentile(Z, 5)] = self.lr_range[1]

        #Update neuron sparsity costs
        sparsity_cost_change = (self.sparsity_cost_range[1] - self.sparsity_cost_range[0])/nblocks
        self.sparsity_cost = self.sparsity_cost + (self.age > 0)*sparsity_cost_change
        self.sparsity_cost[Z < np.percentile(Z, 5)] = self.sparsity_cost_range[0]

        #Update weight mask and weight matrix
        self.w[Z < np.percentile(Z, 5),:] = np.random.normal(0, 0.1, size=np.shape(self.w[Z < np.percentile(Z, 5),:]))
        self.w_mask = np.ones([self.nhidden, self.nvisible])
        #self.w_mask[Z < np.percentile(Z, 5),:] = 0

    def test_performance(self, X):
        '''
            TEST_PERFORMANCE computes the reconstruction error for patterns
             after learning

            INPUTS:
                X = [nvisible X npatterns] input pattern matrix

            OUTPUTS:
                self = rbm object with computed errors prior to training
        '''

        npatterns = np.shape(X)[1]
        self.error['test'] = {
            'd': np.zeros(npatterns),
            'm': np.zeros(npatterns)
        }

        for i in range(npatterns):
            v_data = X[:,i]

            # Up phase
            h_data = self.cd_up(v_data)

            # Down phase
            v_recon, h_recon = self.cd_down(h_data)

            # Compute error
            self.error['test']['d'][i],self.error['test']['m'][i] = self.compute_error(v_data, v_recon)

        self.pattern_separation = self.compute_pattern_separation(X)

    def hamming_distance(self, x, y):
        '''
            HAMMING_DISTANCE computes the hamming distance between two
             binary-valued vectors

            INPUTS:
                x = first binary vector
                y = second binary vector

            OUTPUTS:
                d = Hamming distance between the two vectors
        '''

        d = np.sum(np.abs(x - y))

        return d

    def compute_error(self, v_data, v_recon):
        '''
            COMPUTE_ERROR calculates the reconstruction error
                - Error is calculated as Hamming distance

            INPUTS:
                v_data  = [nvisible X 1] data input vector
                v_recon = [nvisible X 1] reconstructed input vector

            OUTPUTS:
                d = [scalar] Hamming distance between v_data and v_recon
                m = [scalar] percent match between v_data and v_recon
        '''

        d = self.hamming_distance(v_data, v_recon)
        m = 1 - d/self.nvisible

        return d, m

    def compute_pattern_separation(self, X):
        '''
            COMPUTE_PATTERN_SEPARATION calculates the amount of separation
             between two patterns based on the true distance in the data,
             comparing this to the distance in the hidden layer activation

            INPUTS:
                X = [nvisible X npatterns] array of input patterns

            OUTPUTS:
                ps = [npatterns*(npatterns-1)/2] vector of pattern
                      separation statistics, computed as the percent
                      change in hamming distance between the patterns
                      when encoded in the hidden units vs the data.
        '''

        npatterns = np.shape(X)[1]
        ps = {
            'data': np.zeros(int(npatterns*(npatterns-1)/2)),
            'net' : np.zeros(int(npatterns*(npatterns-1)/2))
        }
        ps_index=0
        for i in range(npatterns-1):
            p1 = X[:, i]
            h1 = self.cd_up(p1)
            #v_recon1 = self.cd_down(h1)[0]
            for j in range(i+1, npatterns):
                p2 = X[:, j]
                h2 = self.cd_up(p2)
                #v_recon2 = self.cd_down(h2)[0]

                ps['data'][ps_index] = self.hamming_distance(p1, p2)/(np.sum(p1) + np.sum(p2))
                ps['net'][ps_index]  = self.hamming_distance(h1, h2)/(np.sum(h1) + np.sum(h2))
                ps_index += 1

        return ps

    def train(self, X):
        '''
            TRAIN runs CD-1 algorithm using a batch size of 1.
                - Each patterns is trained with only one iteration to simulate "one-shot learning"

            INPUTS:
                X = [nvisible X npatterns] input pattern matrix

            OUTPUTS:
                self = the rbm object with updated weights
        '''

        npatterns = np.shape(X)[1]
        self.error['d'] = np.zeros(npatterns)
        self.error['m'] = np.zeros(npatterns)

        for i in range(npatterns):
            v_data = X[:,i]

            # Up phase
            h_data = self.cd_up(v_data)

            # Down phase
            v_recon, h_recon = self.cd_down(h_data)

            # Update weights
            dw = np.outer(h_data, v_data) - np.outer(h_recon, v_recon)
            h_sparsity = np.mean(h_data)

            dw = dw *self.w_mask

            lrmtx = matlib.repmat(self.lr, self.nvisible, 1)
            sparsity_cost_mtx = matlib.repmat(self.sparsity_cost, self.nvisible, 1)
            self.w = self.w + lrmtx.T*dw - self.weight_decay*self.w - sparsity_cost_mtx.T*(h_sparsity - self.tgt_sparsity)

            # Compute error
            self.error['d'][i],self.error['m'][i] = self.compute_error(v_data, v_recon)

    def lifespan(self, X, nblocks):
        '''
            LIFESPAN runs training over the input patterns in blocks
             separated by periods of neuronal turnover

            INPUTS:
                X = [nvisible X npatterns] matrix of patterns

        '''

        patterns_per_block = np.floor(np.shape(X)[1]/nblocks)

        self.error['blocks'] = {}

        for i in range(nblocks):
            start_index = int(i*patterns_per_block)
            end_index = int((i+1)*patterns_per_block)
            pattern_block = X[:,start_index:end_index]

            self.train(pattern_block)

            self.error['blocks'][i] = {
                'd': self.error['d'],
                'm': self.error['m']
            }

            if self.neurogenesis is True:
                self.growth(i, nblocks)

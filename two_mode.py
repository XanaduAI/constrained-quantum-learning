# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Perform the two mode constrained variational quantum circuit optimization
to learn the ON state."""
import pickle
import operator
import time
import datetime

import numpy as np
from scipy.optimize import basinhopping

from sklearn.cluster import KMeans

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Sgate


def init(clip_size):
    """Generate an initial random parameter.

    Args:
        clip_size (float): the parameter will be clipped
            to the domain [-clip_size, clipsize].

    Returns:
        float: the random clipped parameter
    """
    return np.random.rand() * 2 * clip_size - clip_size


def on_state(a, cutoff):
    """The ON resource state we would like to learn.

    |psi> = |0> + i*sqrt(3/2)*a|1> + a*i|3>

    Args:
        a (float): the ON state parameter
        cutoff (int): the Fock basis truncation

    Returns:
        array: the density matrix rho=|psi><psi|
    """
    ket = np.zeros([cutoff], dtype=np.complex128)
    ket[0] = 1.
    ket[1] = 1j*np.sqrt(3/2)*a
    ket[3] = 1j*a
    ket = ket/np.linalg.norm(ket)
    dm = np.einsum('i,j->ij', ket, np.conj(ket))
    return dm


def circuit(params, a, m, cutoff):
    """Runs the constrained variational circuit with specified parameters,
    returning the output fidelity to the requested ON state, as well as
    the post-selection probability.

    Args:
        params (list): list of gate parameters for the constrained
            variational quantum circuit. This should contain the following values
            in the following order:

            * ``'sq0' = (r, phi)``: the squeezing magnitude and phase on mode 0
            * ``'disp0' = (r, phi)``: the displacement magnitude and phase on mode 0
            * ``'sq1' = (r, phi)``: the squeezing magnitude and phase on mode 1
            * ``'disp1' = (r, phi)``: the displacement magnitude and phase on mode 1
            * ``'BS' = (theta, phi)``: the beamsplitter angles

        a (float): the ON state parameter
        m (int): the Fock state measurement to be post-selected
        cutoff (int): the Fock basis truncation

    Returns:
        tuple: a tuple containing the output fidelity to the target ON state,
            the probability of post-selection, the state norm before entering the beamsplitter,
            the state norm after exiting the beamsplitter, and the density matrix of the output state.
    """
    # define target state
    ONdm = on_state(a, cutoff)

    # unpack circuit parameters
    sq0_r, sq0_phi, disp0_r, disp0_phi, sq1_r, sq1_phi, disp1_r, disp1_phi, theta, phi = params

    # quantum circuit prior to entering the beamsplitter
    eng1, q1 = sf.Engine(2)
    with eng1:
        Sgate(sq0_r, sq0_phi) | q1[0]
        Dgate(disp0_r, disp0_phi) | q1[0]
        Sgate(sq1_r, sq1_phi) | q1[1]
        Dgate(disp1_r, disp1_phi) | q1[1]

    stateIn = eng1.run('fock', cutoff_dim=cutoff)
    normIn = np.abs(stateIn.trace())

    # norm of output state and probability
    with eng1:
        BSgate(theta, phi) | (q1[0], q1[1])

    stateOut = eng1.run('fock', cutoff_dim=cutoff)
    normOut = np.abs(stateOut.trace())
    rho = stateOut.dm()

    # probability of meausring m1 and m2
    prob = np.abs(np.trace(rho[m, m]))

    # output state
    rhoB = rho[m, m]/prob

    fidelity = np.abs(np.trace(np.einsum('ij,jk->ik', rhoB, ONdm)))
    return (fidelity, prob, normIn, normOut, rhoB)


def loss(params, a, m, cutoff):
    """Returns the loss function of the constrained variational circuit.

    The loss function is given by:

        loss = -fidelity + 10*(1-np.abs(normIn)) + 10*(1-np.abs(normOut))

    Therefore, minimising the loss function will result in the output state
    approaching the target ON state.

    Args:
        params (list): list of gate parameters for the constrained
            variational quantum circuit. This should contain the following values
            in the following order:

            * ``'sq0' = (r, phi)``: the squeezing magnitude and phase on mode 0
            * ``'disp0' = (r, phi)``: the displacement magnitude and phase on mode 0
            * ``'sq1' = (r, phi)``: the squeezing magnitude and phase on mode 1
            * ``'disp1' = (r, phi)``: the displacement magnitude and phase on mode 1
            * ``'BS' = (theta, phi)``: the beamsplitter angles

        a (float): the ON state parameter
        m (int): the Fock state measurement to be post-selected
        cutoff (int): the Fock basis truncation

    Returns:
        float: loss value.
    """
    fidelity, _, normIn, normOut, _ = circuit(params, a, m, cutoff)
    loss = -fidelity + 10*(1-np.abs(normIn)) + 10*(1-np.abs(normOut))
    return loss


def loss_with_prob(params, a, m, cutoff):
    """Returns the loss function of the constrained variational circuit
    with post-selection probability to be also maximised.

    The loss function is given by:

        loss = -fidelity - prob + 10*(1-np.abs(normIn)) + 10*(1-np.abs(normOut))

    Therefore, minimising the loss function will result in the output state
    approaching the target ON state, while also maximising the probability
    of generating the output state.

    Args:
        params (list): list of gate parameters for the constrained
            variational quantum circuit. This should contain the following values
            in the following order:

            * ``'sq0' = (r, phi)``: the squeezing magnitude and phase on mode 0
            * ``'disp0' = (r, phi)``: the displacement magnitude and phase on mode 0
            * ``'sq1' = (r, phi)``: the squeezing magnitude and phase on mode 1
            * ``'disp1' = (r, phi)``: the displacement magnitude and phase on mode 1
            * ``'BS' = (theta, phi)``: the beamsplitter angles

        a (float): the ON state parameter
        m (int): the Fock state measurement to be post-selected
        cutoff (int): the Fock basis truncation

    Returns:
        float: loss value.
    """
    fidelity, prob, normIn, normOut, _ = circuit(params, a, m, cutoff)
    loss = -fidelity -prob + 10*(1-np.abs(normIn)) + 10*(1-np.abs(normOut))
    return loss


def run_global_optimization(a, m, nhp):
    """Run the constrained variational quantum circuit global optimization
    using the basin hopping algorithm.

    Args:
        a (float): the ON state parameter
        m (int): the Fock state measurement to be post-selected
        nhp (int): number of basin hopping iterations

    Returns:
        tuple: optimization results. A tuple of circuit parameters,
            fidelity to the target state, and probability of generating the state.
    """
    # circuit hyperparameters
    clip_size = 1
    cutoff = 15

    # generate the initial parameters
    bound = [clip_size, np.pi]*4+[np.pi]*2
    x0 = map(init, bound)

    # perform the optimization
    minimizer_kwargs = {"method": "SLSQP", "args": (a, m, cutoff)}  # SLSQP L-BFGS-B
    print("hopping....")

    res = basinhopping(loss, list(x0), minimizer_kwargs=minimizer_kwargs, niter=nhp)

    #print the final restuls
    x_f = res.x
    fidelity, prob, _, _, _ = circuit(x_f, a, m, cutoff)
    print("final fid {}, prob {}".format(fidelity, prob))
    return res.x, fidelity, prob


if __name__ == "__main__":
    # Set the optimization hyperparameters
    a = 0.3
    m = 2
    file_name = 'run1'

    print(datetime.datetime.now())
    print("=====================================")
    print("Direct optimization for a = {}, m ={}".format(a, m))
    print("=====================================")
    start = time.time()

    # =============================================================
    # Perform the global optimization
    # =============================================================

    # run the bashin hopping algorithms (each hop nhp times) ntier times
    nhp = 20
    niter = 30

    #store the result from each global search
    dir = 'data'

    hpx = []
    prob_ls = []
    fid_ls = []

    for e in range(niter):
        print("Global explore {}".format(e+1))
        x, fid, prob = run_global_optimization(a, m, nhp)

        prob_ls.append(prob)
        fid_ls.append(fid)
        hpx.append(x)

    # =============================================================
    # remove non-optimal fidelities which will happen occassionally
    # =============================================================

    fid_ls = np.array(fid_ls)
    res = KMeans(n_clusters=2).fit(fid_ls.reshape(-1, 1))
    mean0 = np.mean(fid_ls[np.where(res.labels_ == 0)])
    mean1 = np.mean(fid_ls[np.where(res.labels_ == 1)])

    if np.abs(mean0 - mean1) < 0.01:
        # this means there is no sub-optimal point so we don't need to remove it
        print('Fail')
    else:
        if mean0 > mean1:
            drop = 1
        else:
            drop = 0

        print("mean of cluster 0: {:.3f}, mean of cluster 1: {:.3f}, drop {:d}".format(mean0, mean1, drop))

        prob_ls = np.array(prob_ls)
        prob_ls[np.where(res.labels_ == drop)] = 0
        print(prob_ls)

    # =============================================================
    # Select the best prob
    # NOTE: this is only guaranteed to work if we have the
    # correct optimal fidelity
    # =============================================================

    index, value = max(enumerate(prob_ls), key=operator.itemgetter(1))
    x_opt = hpx[index]
    print("best fid: {}, best prob {}".format(fid_ls[index], prob_ls[index]))

    #save results to file_name.pickle
    with open(file_name + '.pickle', 'wb') as handle:
        pickle.dump(x_opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Runtime: {}".format(time.time()-start))

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
"""Perform the three mode constrained variational quantum circuit optimization
to learn the ON state."""
import pickle
import time
import datetime

import numpy as np
from scipy.optimize import basinhopping, minimize

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


def circuit(params, a, m1, m2, cutoff):
    """Runs the constrained variational circuit with specified parameters,
    returning the output fidelity to the requested ON state, as well as
    the post-selection probability.

    Args:
        params (list): list of gate parameters for the constrained
            variational quantum circuit. This should contain the following 15 values
            in the following order:

            * ``sq_r0, sq_r1, sq_r2``: the squeezing magnitudes applied to the first three modes
            * ``sq_phi0, sq_phi1, sq_phi2``: the squeezing phase applied to the first three modes
            * ``d_r0, d_r1, d_r2``: the displacement magnitudes applied to the first three modes
            * ``bs_theta1, bs_theta2, bs_theta3``: the 3-mode interferometer beamsplitter angles theta
            * ``bs_phi1, bs_phi2, bs_phi3``: the 3-mode interferometer beamsplitter phases phi

        a (float): the ON state parameter
        m1 (int): the Fock state measurement of mode 0 to be post-selected
        m2 (int): the Fock state measurement of mode 1 to be post-selected
        cutoff (int): the Fock basis truncation

    Returns:
        tuple: a tuple containing the output fidelity to the target ON state,
            the probability of post-selection, the state norm before entering the beamsplitter,
            the state norm after exiting the beamsplitter, and the density matrix of the output state.
    """
    # define target state
    ONdm = on_state(a, cutoff)

    # unpack circuit parameters
    # squeezing magnitudes
    sq_r = params[:3]
    # squeezing phase
    sq_phi = params[3:6]
    # displacement magnitudes (assume displacement is real for now)
    d_r = params[6:9]
    # beamsplitter theta
    bs_theta1, bs_theta2, bs_theta3 = params[9:12]
    # beamsplitter phi
    bs_phi1, bs_phi2, bs_phi3 = params[12:]

    # quantum circuit prior to entering the beamsplitter
    prog = sf.Program(3)

    with prog.context as q:
        for k in range(3):
            Sgate(sq_r[k], sq_phi[k]) | q[k]
            Dgate(d_r[k]) | q[k]

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    stateIn = eng.run(prog).state
    normIn = np.abs(stateIn.trace())

    # norm of output state and probability
    prog_BS = sf.Program(3)
    with prog_BS.context as q:
        BSgate(bs_theta1, bs_phi1) | (q[0], q[1])
        BSgate(bs_theta2, bs_phi2) | (q[1], q[2])
        BSgate(bs_theta3, bs_phi3) | (q[0], q[1])

    stateOut = eng.run(prog_BS).state
    normOut = np.abs(stateOut.trace())
    rho = stateOut.dm()

    # probability of meausring m1 and m2
    prob = np.abs(np.trace(rho[m1, m1, m2, m2]))

    # output state
    rhoC = rho[m1, m1, m2, m2]/prob

    #fidelity with the target
    fidelity = np.abs(np.trace(np.einsum('ij,jk->ik', rhoC, ONdm)))
    return (fidelity, prob, normIn, normOut, rhoC)


def loss(params, a, m1, m2, cutoff):
    """Returns the loss function of the constrained variational circuit.

    The loss function is given by:

        loss = -fidelity + 10*(1-np.abs(normIn)) + 10*(1-np.abs(normOut))

    Therefore, minimising the loss function will result in the output state
    approaching the target ON state.

    Args:
        params (list): list of gate parameters for the constrained
            variational quantum circuit. This should contain the following 15 values
            in the following order:

            * ``sq_r0, sq_r1, sq_r2``: the squeezing magnitudes applied to the first three modes
            * ``sq_phi0, sq_phi1, sq_phi2``: the squeezing phase applied to the first three modes
            * ``d_r0, d_r1, d_r2``: the displacement magnitudes applied to the first three modes
            * ``bs_theta1, bs_theta2, bs_theta3``: the 3-mode interferometer beamsplitter angles theta
            * ``bs_phi1, bs_phi2, bs_phi3``: the 3-mode interferometer beamsplitter phases phi

        a (float): the ON state parameter
        m1 (int): the Fock state measurement of mode 0 to be post-selected
        m2 (int): the Fock state measurement of mode 1 to be post-selected
        cutoff (int): the Fock basis truncation

    Returns:
        float: loss value.
    """
    fidelity, _, normIn, normOut, _ = circuit(params, a, m1, m2, cutoff)
    loss = -fidelity + 10 * (1 - np.abs(normIn)) + 10 * (1 - np.abs(normOut))
    return loss


def loss_with_prob(params, a, m1, m2, cutoff):
    """Returns the loss function of the constrained variational circuit
    with post-selection probability to be also maximised.

    The loss function is given by:

        loss = -fidelity - prob + 10*(1-np.abs(normIn)) + 10*(1-np.abs(normOut))

    Therefore, minimising the loss function will result in the output state
    approaching the target ON state, while also maximising the probability
    of generating the output state.

    Args:
        params (list): list of gate parameters for the constrained
            variational quantum circuit. This should contain the following 15 values
            in the following order:

            * ``sq_r0, sq_r1, sq_r2``: the squeezing magnitudes applied to the first three modes
            * ``sq_phi0, sq_phi1, sq_phi2``: the squeezing phase applied to the first three modes
            * ``d_r0, d_r1, d_r2``: the displacement magnitudes applied to the first three modes
            * ``bs_theta1, bs_theta2, bs_theta3``: the 3-mode interferometer beamsplitter angles theta
            * ``bs_phi1, bs_phi2, bs_phi3``: the 3-mode interferometer beamsplitter phases phi

        a (float): the ON state parameter
        m1 (int): the Fock state measurement of mode 0 to be post-selected
        m2 (int): the Fock state measurement of mode 1 to be post-selected
        cutoff (int): the Fock basis truncation

    Returns:
        float: loss value.
    """
    fidelity, prob, normIn, normOut, _ = circuit(params, a, m1, m2, cutoff)
    loss = -fidelity - prob + 10 * (1 - np.abs(normIn)) + 10 * (1 - np.abs(normOut))
    return loss


i = 0
xf = []
fid_progress = []
prob_progress = []


class stopException(Exception):
    """Exception used if the optimization is stopped"""
    pass


def run_global_optimization(a, m1, m2, dir='data'):
    """Run the constrained variational quantum circuit global optimization
    using the basin hopping algorithm.

    Args:
        a (float): the ON state parameter
        m1 (int): the Fock state measurement of mode 0 to be post-selected
        m2 (int): the Fock state measurement of mode 1 to be post-selected
        dir (str): data directory to save output

    Returns:
        tuple: optimization results. A tuple of circuit parameters,
            fidelity to the target state, and probability of generating the state.
    """
    # circuit hyperparameters
    clip_size = 1
    cutoff = 15

    # generate the initial parameters
    bound = [clip_size] * 3 + [np.pi] * 3 + [clip_size] * 3  + [np.pi] * 6
    x0 = map(init, bound)

    # perform the optimization
    minimizer_kwargs = {"method": "SLSQP", "args": (a, m1, m2, cutoff)}  # SLSQP L-BFGS-B

    def myAccept(xk, f, accepted):
        """ this accept condition is to save some computational time which is optional"""
        global i
        global fid_progress
        global prob_progress
        i = i + 1
        fidelity, prob, normIn, normOut, _ = circuit(xk, a, m1, m2, cutoff)
        fid_progress.append(fidelity)
        prob_progress.append(prob)

        print("Hopping {} fidelity: {}, prob: {}, normIn: {}, normOut: {}".format(i, fidelity, prob, round(normIn, 2), round(normOut, 2)))
        if fidelity >= .9999 and prob >= 1e-4:
            file_name = dir
            results = {'para': xk,
                       'fid_progress': fid_progress,
                       'prob_progress': prob_progress}
            with open(file_name + '.pickle', 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(time.time() - start)
            raise stopException()

    print("optimizing a={}".format(a))

    res = basinhopping(loss, list(x0), minimizer_kwargs=minimizer_kwargs, niter=40, callback=myAccept)

    fidelity, prob, _, _, _ = circuit(res.x, a, m1, m2, cutoff)
    print(fidelity, prob)

    return res.x, fidelity, prob


def run_local_optimization(a, m1, m2, init_params):
    """Run the constrained variational quantum circuit global optimization
    using the basin hopping algorithm.

    Args:
        a (float): the ON state parameter
        m1 (int): the Fock state measurement of mode 0 to be post-selected
        m2 (int): the Fock state measurement of mode 1 to be post-selected
        init_params (Sequence): initial gate parameters

    Returns:
        tuple: optimization results. A tuple of circuit parameters,
            fidelity to the target state, and probability of generating the state.
    """
    cutoff = 15

    def printfunc(xk):
        """Callback print function for the BFGS minimisation algorithm"""
        global i
        global xf
        # global fidelity
        # global prob
        i = i+1
        fidelity, prob, normIn, normOut, _ = circuit(xk, a, m1, m2, cutoff)

        xf = xk
        # fidelity = fidelity_n
        # prob = prob_n
        print("step {} fidelity: {}, prob: {}, normIn: {}, normOut: {}".format(i, fidelity, prob, normIn, normOut))
        if  fidelity >= .999 and prob >= 0.02:
            raise stopException()

    # run optimization
    args = [a, m1, m2, cutoff]
    try:
        res = minimize(loss_with_prob, init_params, args=args, method='BFGS', callback=printfunc)
    except stopException:
        out_file_name = 'data/final3mode' + 'a=' + str(a) + 'm1=' + str(m1) + 'm2=' + str(m2)

        with open(out_file_name + '.pickle', 'wb') as handle:
            pickle.dump(xf, handle, protocol=pickle.HIGHEST_PROTOCOL)

        fid, prob, _, _, _ = circuit(xf, a, m1, m2, cutoff)
        print("Final fidelity: {}, prob: {}".format(fid, prob))

    return res.x, fid, prob


if __name__ == "__main__":
    # hyperparameters
    a = 0.3
    m1 = 1
    m2 = 2
    global_save_file_name = 'run1_global'

    print(datetime.datetime.now())
    print("==============")
    print("Basin Hopping")
    print("==============")

    start = time.time()

    x0, _, _ = run_global_optimization(a, m1, m2, global_save_file_name)
    time_elapsed = time.time() - start
    print('Runtime: ', time_elapsed)

    start = time.time()
    print(datetime.datetime.now())
    print("==============")
    print("Local search for high prob")
    print("==============")

    i = 0
    xf = []

    run_local_optimization(a, m1, m2, x0)
    print("Use time {}".format(time.time()-start))

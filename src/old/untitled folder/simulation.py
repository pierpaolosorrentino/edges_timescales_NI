import os
import numpy as np
from thetanmm.model import MontbrioPazoRoxin
from tvb.simulator.lab import *
from tvb.basic.neotraits.api import HasTraits, Attr, Final

import itertools

def configure_sim(
        dataset, 
        G, 
        cut_in=10000, 
        sim_len=600e3, 
        conn_speed=2.0,
        dt=0.01,
        tavg=True,
        bold=True,
        eeg=False,
        raw=False,
        seed=42,
        nsigma=0.01,
        eta=-4.6,
        J=14.5,
        Delta=0.7,
        tau=1,
        initial_conditions=None,
        mult_noise_eq=None
):
    """ Configures simulation with Monbrio model

    Parameters
        dataset        dictionary of TVB components; obligatory:`connectivity`, other: `eeg`
        G              global coupling scaling
        cut_in         length [ms] of the discarded initial transient
        sim_len        length [ms] of simulation
        conn_speed     connection speed 
        dt             integration step
        tavg           include time average monitor
        bold           include BOLD monitor
        eeg            include EEG monitor, requires the monitor to be provided in the dataset
        raw            include RAW monitor
        seed           random number generator seed
        nsigma         sigma for the noise
        eta            Montbrio model parameter
        J              Montbrio model parameter
        Delta          Montbrio model parameter

    Returns:
        sim            configured simulator instance
    """

    conn = dataset["connectivity"]
    conn.speed = np.array([conn_speed])  
    np.fill_diagonal(conn.weights, 0.)
    conn.weights = conn.weights/np.max(conn.weights)
    conn.configure()

    mont_model = MontbrioPazoRoxin(
            eta   = np.r_[eta],
            J     = np.r_[J],
            Delta = np.r_[Delta],
            tau = np.r_[tau],
    )
    con_coupling = coupling.Scaling(a=np.array([G]))

    nsig = np.r_[nsigma, nsigma*2] if np.isscalar(nsigma) else nsigma

    if mult_noise_eq is not None:
        hiss = noise.Multiplicative(
                nsig=nsig, 
                noise_seed=seed,
                b=mult_noise_eq
        )
    else:
        hiss = noise.Additive(nsig=nsig, noise_seed=seed)

    integrator = integrators.HeunStochastic(dt=dt, noise=hiss)



    mntrs = []
    if tavg:
        mntrs.append(monitors.TemporalAverage(period=1.))
    if bold:
        BOLD_period=2000
        mntrs.append( monitors.Bold(period=BOLD_period) )
    if eeg:
        eeg_period = 1000/256.
        eeg_monitor = dataset["eeg"]
        eeg_monitor.period = eeg_period
        mntrs.append(eeg_monitor)
    if raw:
        mntrs.append(monitors.Raw())


    sim = simulator.Simulator(model=mont_model,
                              connectivity=conn,
                              coupling=con_coupling,
                              conduction_speed=conn_speed,
                              integrator=integrator,
                              monitors=mntrs,
                              simulation_length=sim_len + cut_in,
                              initial_conditions=initial_conditions
    )
    sim.configure()

    return sim

def generate_initial_conditions_array(sim):
    sim.connectivity.set_idelays(dt=sim.integrator.dt)
    horizon = sim.horizon
    nvar = sim.model.nvar
    nnodes = sim.number_of_nodes
    nmodes = sim.model.number_of_modes

    return np.zeros( (horizon, nvar, nnodes, nmodes) )

def generate_rescaled_initial_conditions(sim, state_variable_range):
    """
    Parameters
    sim: pre-configured simulator instance (use sim.configure())
    state_variable_range: ranges to be used instead of `model.state_variable_range`

    """
    sim.connectivity.set_idelays(dt=sim.integrator.dt)
    horizon = sim.horizon
    nvar = sim.model.nvar
    nnodes = sim.number_of_nodes
    nmodes = sim.model.number_of_modes

    initial_conditions = generate_initial_conditions_array(sim)

    #var_shape = ( horizon, 1, nnodes, nmodes)
    var_shape = list(initial_conditions.shape)
    var_shape[1] = 1
    for i, var in enumerate(sim.model.state_variables):
        low, high = state_variable_range[var]
        initial_conditions[:,[i],:,:] = np.random.uniform(low=low, high=high, size=(var_shape))

    return initial_conditions

def rescale_nsigma(sig, tau):
    sig_r = sig * 1/(tau**4)
    sig_V = 2 * sig * 1/(tau**2)
    return  np.r_[sig_r, sig_V]


def save_sim_output(sim, outputs, out_path, reduced_precision=True):
    keys = [(m.__class__.__name__ + "_time", m.__class__.__name__ + "_data") for m in sim.monitors]
    flat_outs = dict(
            zip(
                itertools.chain(*keys),
                itertools.chain(*outputs)
            )
    )
    if reduced_precision:
        for k,val in flat_outs.items():
            flat_outs[k] = val.astype(np.float32)
    np.savez( out_path, **flat_outs )

def run_sim(sim, out_path, cut_in, sim_len):
    if cut_in > 0:
        _ = sim.run(simulation_length=cut_in)

    outputs = sim.run(simulation_length=sim_len)
    save_sim_output(sim, outputs, out_path)

def run_sim_variable_parameter(sim, param_vals, sim_len):
    """
    sim:        simulator instance
    param_vals: values for variable parameters {var: vals} where `vals` is an
                array of len `n` for all `var`
    sim_len:    total length of simulation (will be divided into `n` segments
    """
    output = []
    for _ in sim.monitors:
        output.append( ([], []) )


    n_segs = len(list(param_vals.values())[0])
    seg_len = sim_len / n_segs

    for n in range(n_segs):
        for var, vals in param_vals.items():
            setattr(sim.model, var, np.r_[vals[n]])

        partial_output = sim.run(simulation_length=seg_len)

        for j, (times,data) in enumerate(partial_output):
            output[j][0].append(times)
            output[j][1].append(data)


    np_output = []
    for time, data in output:
        np_output.append( ( np.concatenate(time), np.concatenate(data)))

    return np_output

def generate_Js(sigma_J, N, J0):
    mu_J = sigma_J * np.sqrt(2/np.pi)
    return J0 + np.abs(np.random.normal(scale=sigma_J,size=N))- mu_J


class LinearNorm(equations.TemporalApplicableEquation):
    equation = Final(
        label="Equation",
        default="a * abs(x_0 - var) + b",
        doc=""":math:`result = a * (|x_0-x|) + b`""")

    parameters = Attr(
        field_type=dict,
        label="Parameters",
        default=lambda: {"a": 1.0, "b": 0.0, "x_0":0})

def MontbrioPazoRoxin_up_state(tau, Delta, J, eta):
    r_0 = sorted(
            np.roots(
                p=np.r_[
                    tau**2 * np.pi**2,
                    -J * tau,
                    -eta,
                    Delta/(4*np.pi**2*tau**2),
                    0
                ]
            )
    )[-1]
    V_0 = -Delta/(2*np.pi*tau*r_0)

    return r_0, V_0




    

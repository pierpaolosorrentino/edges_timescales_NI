import numpy as np
import itertools
from tvb.simulator.lab import *

def save_sim_output(sim, outputs, out_path, reduced_precision=True):
    """Save simulated time series in npz format under keywords corresponding to
    monitor names.

    Parameters
    ----------
    sim : tvb.simulator
        Instance of the simulator which produced the outputs.
    outputs : list 
        Return value of sim.run(): list of pairs of time and data arrays. 
    out_path : str
        Path to the npz file.
    reduced_precision : bool
        Save in float32 precision.

    Examples
    --------
    >>> outputs = sim.run(simulation_length=600e3)
    >>> save_sim_output(sim, outputs, 'simulated_ts.npz')
    >>> data = np.load('simulated_ts.npz')
    >>> data.files
    ['TemporalAverage_time', 'TemporalAverage_data', 'Bold_time', 'Bold_data']
    """
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
    
def return_output(sim, outputs, reduced_precision=True):
    """Save simulated time series in npz format under keywords corresponding to
    monitor names.

    Parameters
    ----------
    sim : tvb.simulator
        Instance of the simulator which produced the outputs.
    outputs : list 
        Return value of sim.run(): list of pairs of time and data arrays. 
    reduced_precision : bool
        Save in float32 precision.

    Examples
    --------
    >>> outputs = sim.run(simulation_length=600e3)
    >>> save_sim_output(sim, outputs, 'simulated_ts.npz')
    >>> data = np.load('simulated_ts.npz')
    >>> data.files
    ['TemporalAverage_time', 'TemporalAverage_data', 'Bold_time', 'Bold_data']
    """
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
            
    return flat_outs 

def integrate_decoupled_nodes(
    model,
    N=1,
    nsteps=1,
    deterministic=False,
    noise_sigma=0.01,
    dt=0.01,
    initial_conditions=None,
    tau = 1
):
    """Integrate decoupled nodes.

    Parameters
    ----------
    model : tvb.simulator.model
        configured instance of the model
    N : int
        number of nodes to integrate
    nsteps : int
        number of steps
    deterministic : bool
        selects integrator; True for Runge-Kutta noise, False for Heun with noise
    noise_sigma : float
        sets the noise sigma if deterministic=False
    dt : float
        integration step
    initial_conditions : np.array
        initial conditions of shape (1, model.nvar, N, model.number_of_modes)
    """

    def buffer_shape(model, steps, n_nodes):
        return steps, model.nvar, n_nodes, model.number_of_modes

    model.configure()
    
    if deterministic:
        integrator = integrators.RungeKutta4thOrderDeterministic(dt=dt)
    else:
        hiss = noise.Additive(
            nsig=np.r_[noise_sigma/tau,noise_sigma*2].reshape(2,1,1)
        )
        integrator = integrators.HeunStochastic(dt=dt, noise=hiss)
        integrator.noise.configure_white(dt, buffer_shape(model, 1,N))


    if model.state_variable_boundaries is not None:
        indices = []
        boundaries = []
        for sv, sv_bounds in model.state_variable_boundaries.items():
            indices.append(model.state_variables.index(sv))
            boundaries.append(sv_bounds)
        sort_inds = np.argsort(indices)
        integrator.bounded_state_variable_indices = np.array(indices)[sort_inds]
        integrator.state_variable_boundaries = np.array(boundaries).astype("float64")[sort_inds]
    else:
        integrator.bounded_state_variable_indices = None
        integrator.state_variable_boundaries = None
    
    trajectory = np.zeros( 
        buffer_shape(model, nsteps+1, N)
    )
    if initial_conditions is None:
        initial_conditions = model.initial(
            dt=dt, 
            history_shape=buffer_shape(model, 1, N)
        )
    trajectory[0,:] = initial_conditions
    
    no_coupling = np.zeros( (model.nvar, 1, model.number_of_modes) )
    state = trajectory[0,:]
    
    for i in range(nsteps):
        state = integrator.scheme(
            state,
            model.dfun,
            coupling=no_coupling,
            local_coupling=0.0,
            stimulus=0.0
        )
        trajectory[i+1, :] = state 
        
    return trajectory

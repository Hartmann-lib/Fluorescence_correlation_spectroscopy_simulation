import numpy as np

"""
27.09.2021 | Andreas Hartmann
"""

def trajectory_2states(k_on, k_off, DT, N_STEPS):
    """
    State trajectory of 2 states with interconversion rates k_on and k_off
    
    k_on : (ms^-1) interconversion rate 0->1
    k_off : (ms^-1) interconversion rate 1->0
    DT : (µs) time step size
    N_STEPS : number of time steps
    
    return state trajectory
    """

    # estimating number of dwell times to draw
    numR = int(np.round(DT*N_STEPS/1000*np.max([k_on, k_off])*10))

    # drawing dwell times for each state
    dwellA = np.random.exponential(1000/k_off, numR) # (µs)
    dwellD = np.random.exponential(1000/k_on, numR) # (µs)

    # interweaving dwell times
    dwells = np.ravel((dwellA, dwellD), order='F') # (µs)

    # calculate transition times
    t_trans = np.cumsum(np.insert(dwells, 0, 0)) # (µs)
    trace = np.ravel((t_trans[0:-1] - 0.5*DT, t_trans[1:] + 0.5*DT), order='F') # (µs)

    # assign states to the time points
    stateA = np.ones(numR)
    stateD = np.zeros(numR)
    states = np.ravel((stateA, stateD), order='F')
    trace_states = np.ravel((states, states), order='F')

    t = np.arange(0, N_STEPS, 1)*DT # (µs)

    return np.interp(t, trace, trace_states)

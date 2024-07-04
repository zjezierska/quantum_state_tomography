import numpy as np
import argparse
import qutip as qt
import os
from src.config import DIMS, NUM_SAMPLES, TRAJ_LENGTH, NOF_SAMPLES_DISTR, OUTPUT_DIR, ALPHA, GAMMA, a, x, p, H_QUARTIC, H_HARMONIC, c_ops, X_LIST, EVO_DIMS, T_LIST
from src.utils.quantum_utils import calculate_psi_products_simple

def generate_states(dims, num_samples):
    """
    Generate quantum states and return the data.
    
    Parameters:
    - dims (int): The dimension of the quantum state.
    - num_samples (int): The number of samples to generate.
    
    Returns:
    - ndarray: An array of shape (num_samples, dims, dims) containing the generated quantum states.
    """
    
    # Generate random density matrices
    rho_list = [qt.rand_dm(dims) for _ in range(num_samples)]
    
    return np.array([rho.full() for rho in rho_list])

def calculate_probabilities(rho_matrix):
    """Calculate probabilities based on the state and psi products."""
    psi_products = calculate_psi_products_simple(X_LIST, DIMS)
    interaction = rho_matrix[:, :, np.newaxis] * psi_products # P(x) = sum ( rho * psi_products ) by definition
    return list(np.sum(np.sum(interaction.real, axis=0), axis=0))

def validate_evolved_state(rho_matrix):
    """Ensure the state's last column and row sums are within acceptable limits."""
    if rho_matrix[:, -1].sum() > 1e-5 or rho_matrix[-1, :].sum() > 1e-5:
        raise ValueError("Sum of the last column or row is not negligible.")

# generate trajectories for the given quantum states using qt.mesolve
def evolve_state_and_get_trajectory(H, initial_state):
    result = qt.mesolve(H, initial_state, T_LIST, c_ops=c_ops)
    trajectory = []

    for _, state in enumerate(result.states):
        rho_matrix = state.full()
        validate_evolved_state(rho_matrix)
        prob_list = calculate_probabilities(rho_matrix)
        trajectory.append(prob_list)
    
    return trajectory

def generate_state_targets_and_trajectories(H, generated_states):
    """
    Generate quantum state trajectories and return the data.

    Parameters:
    - H (Hamiltonian): The Hamiltonian used for the trajectory evolution.
    - generated_states (ndarray): The array of shape of shape (num_samples, dims, dims) of generated quantum states.

    Returns:
    - targets (ndarray): Array of targets for the model (original states - flattened).
    - trajectories (ndarray): Array of generated trajectories.
    """
    targets = []
    trajectories = []
    
    num_samples = len(generated_states)

    for i in range(num_samples):
        
        #short preparation of the initial state
        full_array = np.zeros((EVO_DIMS, EVO_DIMS), dtype=np.complex128)
        full_array[:DIMS, :DIMS] = generated_states[i]
        full_state = qt.Qobj(full_array)
        
        # generate the target for model
        targets.append(np.concatenate((generated_states[i].real.flatten(), generated_states[i].imag.flatten())))
        
        # generate the trajectory
        trajectory = evolve_state_and_get_trajectory(H, full_state)

        # append it to the samples list
        trajectories.append(np.array(trajectory))
    
    return np.array(targets), np.array(trajectories)

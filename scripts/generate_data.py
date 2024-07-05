import numpy as np
import argparse
import qutip as qt
import os
from src.config import DIMS, EVO_DIMS, NUM_SAMPLES, BATCH_SIZE_GEN, TRAJ_LENGTH, NOF_SAMPLES_DISTR, OUTPUT_DIR, X_MAX, ALPHA
from src.utils.quantum_utils import calculate_psi_products_simple

def generate_states(dims, num_samples_to_generate):
    """
    Generate quantum states and return the data.
    
    Parameters:
    - dims (int): The dimension of the quantum state.
    - num_samples_to_generate (int): The number of samples to generate.
    
    Returns:
    - ndarray: An array of shape (num_samples_to_generate, dims, dims) containing the generated quantum states.
    """
    
    # Generate random density matrices
    rho_list = [qt.rand_dm(dims) for _ in range(num_samples_to_generate)]
    
    return np.array([rho.full() for rho in rho_list])

def calculate_probabilities(rho_matrix, psi_products):
    """
    Calculate probabilities based on the state and psi products.

    Parameters:
    rho_matrix (ndarray): The density matrix representing the quantum state.

    Returns:
    list: A list of probabilities corresponding to each measurement outcome.
    """
    interaction = rho_matrix[:, :, np.newaxis] * psi_products # P(x) = sum ( rho * psi_products ) by definition
    return list(np.sum(np.sum(interaction.real, axis=0), axis=0))

def validate_evolved_state(rho_matrix):
    """Ensure the state's last column and row sums are within acceptable limits."""
    try:
        if rho_matrix[:, -1].sum() > 1e-5 or rho_matrix[-1, :].sum() > 1e-5:
            raise ValueError("Sum of the last column or row is not negligible.")
    except ValueError as e:
        print(f"Error: {e}")

# generate trajectories for the given quantum states using qt.mesolve
def evolve_state_and_get_trajectory(initial_state, psi_products, t_list, hamiltonian, c_ops=[]):
    """
    Evolves the initial state according to the given Hamiltonian and returns the trajectory of probabilities.

    Parameters:
    - H (qutip.Qobj): The Hamiltonian operator describing the system dynamics.
    - initial_state (qutip.Qobj): The initial state of the system.

    Returns:
    - trajectory (list): A list of probability lists representing the trajectory of the system.

    """
    result = qt.mesolve(hamiltonian, initial_state, t_list, c_ops=c_ops)
    trajectory = []

    for _, state in enumerate(result.states):
        rho_matrix = state.full()
        validate_evolved_state(rho_matrix)
        prob_list = calculate_probabilities(rho_matrix, psi_products)
        trajectory.append(prob_list)
    
    return trajectory

def generate_trajectories(dims, evo_dims, generated_states, psi_products, t_list, hamiltonian, c_ops):
    """
    Generate quantum state trajectories and return the data.

    Parameters:
    - H (Hamiltonian): The Hamiltonian used for the trajectory evolution.
    - generated_states (ndarray): The array of shape of shape (num_samples, dims, dims) of generated quantum states.

    Returns:
    - targets (ndarray): Array of targets for the model (original states - flattened).
    - trajectories (ndarray): Array of generated trajectories.
    """
    
    trajectories = []
    
    num_generated_samples = len(generated_states)

    for i in range(num_generated_samples):
        
        #short preparation of the initial state
        full_array = np.zeros((evo_dims, evo_dims), dtype=np.complex128)
        #check if dims equals generated_states shape
        assert dims == generated_states[i].shape[0], f"Dimensions mismatch: {dims} != {generated_states[i].shape[0]}"
        full_array[:dims, :dims] = generated_states[i]
        full_state = qt.Qobj(full_array)
        
        # generate the trajectory
        trajectory = evolve_state_and_get_trajectory(full_state, psi_products, t_list, hamiltonian, c_ops)

        # append it to the samples list
        trajectories.append(np.array(trajectory))
    
    return np.array(trajectories)

def save_data(states, trajectories, output_dir, batch_num):
    """
    Save the generated states and trajectories to .npy files.

    Parameters:
    states (np.ndarray): Array of quantum states.
    trajectories (np.ndarray): Array of trajectories.
    output_dir (str): Directory to save the .npy files.
    batch_num (int): The batch number.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(os.path.join(output_dir, f'states_batch_{batch_num}.npy'), states)
    np.save(os.path.join(output_dir, f'trajectories_batch_{batch_num}.npy'), trajectories)

def generate_and_save_data(dims, evo_dims, num_samples, batch_size, traj_length, nof_samples_distr, output_dir, hamiltonian, gamma=0.0, alpha=0.1):
    """
    Generate quantum states and trajectories and save the data in batches.

    Parameters:
    - dims (int): The dimension of the quantum state.
    - num_samples (int): The number of samples to generate.
    - batch_size (int): The size of each batch.
    - output_dir (str): Directory to save the .npy files.
    - H (qutip.Qobj): The Hamiltonian operator describing the system dynamics.
    """
    
    a = qt.destroy(dims) # Annihilation operator
    x = a.dag() + a # Position operator
    p = 1j * (a.dag() - a) # Momentum operator
    
    if hamiltonian == "H_HARMONIC":
        print("Using the Harmonic Hamiltonian, setting the quartic parameter to 0...")
        h = a.dag() * a
    elif hamiltonian == "H_QUARTIC":
        print(f"Using the Quartic Hamiltonian with alpha = {alpha}...")
        h = p**2 / 4 + (x / alpha)**4 # Quartic potential Hamiltonian
    else:
        raise ValueError("Invalid Hamiltonian. Please choose either 'H_HARMONIC' or 'H_QUARTIC'.")
    
    if gamma > 0:
        print(f"Using decoherence with gamma = {gamma}...")
        c_ops = [np.sqrt(gamma) * x]
        
    
    t_list = np.linspace(0, 2 * np.pi, traj_length) # Time list for the trajectory
    x_list = np.linspace(-X_MAX, X_MAX, nof_samples_distr) # Position list for psi products
    psi_products = calculate_psi_products_simple(x_list, dims) # Calculate psi products for the given dimensions (simple version)
    
    if num_samples < batch_size:
        print("Number of samples is less than the batch size. Generating all samples in one batch...")
        states = generate_states(dims, num_samples)
        trajectories = generate_trajectories(dims, evo_dims, states, psi_products, t_list, h, c_ops)
        save_data(states, trajectories, output_dir, 0)
        print("Data generation completed.")
        return None
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        print(f"Generating batch {i + 1}/{num_batches}...")
        states = generate_states(dims, batch_size)
        trajectories = generate_trajectories(dims, evo_dims, states, psi_products, t_list, h, c_ops)
        save_data(states, trajectories, output_dir, i)
        del states, trajectories

    # generate with batch size until possible, make the last batch with the remaining samples
    if num_samples % batch_size != 0:
        print(f"Generating batch {num_batches + 1}/{num_batches}...")
        states = generate_states(dims, num_samples % batch_size)
        trajectories = generate_trajectories(dims, evo_dims, states, psi_products, t_list, h, c_ops)
        save_data(states, trajectories, output_dir, num_batches)
        del states, trajectories
    
    print("Data generation completed.")
        

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate quantum states and trajectories.")
    parser.add_argument('--dims', type=int, default=DIMS, help='Dimension of the quantum states.')
    parser.add_argument('--evo_dims', type=int, default=EVO_DIMS, help='Evolution dimension of the quantum states.')
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES, help='Number of samples to generate.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_GEN, help='Size of each batch.')
    parser.add_argument('--traj_length', type=int, default=TRAJ_LENGTH, help='Length of each trajectory.')
    parser.add_argument('--nof_samples_distr', type=int, default=NOF_SAMPLES_DISTR, help='Number of position samples to sample from the distribution.')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory to save the generated .npy files.')
    parser.add_argument('--Hamiltonian', type=str, default="H_HARMONIC", help='Hamiltonian to use for the trajectory evolution.')
    parser.add_argument('--decoherence', type=float, default=0.0, help='Decoherence factor between 0 and 1.')
    parser.add_argument('--quarticity', type=float, default=ALPHA, help='Parameter for the quartic potential Hamiltonian.')
    
    args = parser.parse_args()
    
    generate_and_save_data(args.dims, args.evo_dims, args.num_samples, args.batch_size, args.traj_length, args.nof_samples_distr, args.output_dir, args.Hamiltonian, args.decoherence, args.quarticity)
    
if __name__ == "__main__":
    main()
    
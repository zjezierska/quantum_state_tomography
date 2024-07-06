import qutip as qt
import numpy as np
import math
from scipy.special import hermite

def give_back_matrix(vectr):
    """
    Converts a vector representation of a quantum state into a density matrix.

    Parameters:
    vectr (numpy.ndarray): The vector representation of the quantum state.

    Returns:
    qutip.Qobj: The density matrix representation of the quantum state.
    """

    global d

    # Reshape the vector into a 2D array with real and imaginary parts
    vec = vectr.reshape(2, d**2)

    # Combine the real and imaginary parts to create a complex matrix
    matrix = vec[0, :] + 1j * vec[1, :]

    # Reshape the matrix to have dimensions (d, d)
    matrix = matrix.reshape(d, d)

    # Create a Qobj using the reshaped matrix
    return qt.Qobj(matrix)

def my_fidelity(vec1, vec2):
    """
    Calculate the fidelity between two quantum states in the Fock basis.

    Parameters:
    vec1 (numpy.ndarray): The first quantum state vector.
        A 1-dimensional numpy array representing the quantum state.
    vec2 (numpy.ndarray): The second quantum state vector.
        A 1-dimensional numpy array representing the quantum state.

    Returns:
    float: The fidelity between vec1 and the normalized vec2.

    Raises:
    ValueError: If vec1 is not Hermitian.
    """
    
    # Convert input vectors to Qobj matrices
    vec1 = give_back_matrix(vec1)
    vec2 = give_back_matrix(vec2)

    # Check if vec1 is Hermitian
    if vec1.isherm:
        # Normalize vec2
        vec2_normalized = (vec2.dag() * vec2) / (vec2.dag() * vec2).tr()

        # Calculate and return the fidelity between vec1 and the normalized vec2
        return qt.fidelity(vec1, vec2_normalized)
    else:
        raise ValueError('X is not Hermitian!')

def calculate_psi_products_simple(position_array, dim):    
    """
    Calculate the product of wave functions for a given position array and dimension.

    Parameters:
    position_array (array-like): List of position values.
    dim (int): Dimension of the wave functions.

    Returns:
    array-like: Product matrix of wave functions.

    """
    exp_list = np.exp(-position_array**2 / 2.0)
    norm_list = [np.pi**(-0.25) / np.sqrt(2.0**m * math.factorial(m)) for m in range(dim)]
    herm_list = [np.polyval(hermite(m), position_array) for m in range(dim)]
    
    product_matrix = np.array([
        exp_list * norm_list[m] * herm_list[m] * exp_list * norm_list[n] * herm_list[n]
        for m in range(dim) for n in range(dim)
    ]).reshape((dim, dim, -1))
    return product_matrix

def calculate_psi_products_theta(position_array: np.ndarray, theta_list: np.ndarray, dim: int) -> np.ndarray:
    """
    Calculate the product matrix of positional psi functions for given position_array, theta_list, and dimension.

    Parameters:
    position_array (array-like): 2D array of position values for which to calculate the psi functions.
    theta_list (array-like): List of theta values for which to calculate the psi functions.
    dim (int): Dimension of the product matrix.

    Returns:
    array-like: Product matrix of psi functions with shape (len(theta_list), len_x, dim, dim).
    """
    assert position_array.shape[0] == len(theta_list), "position_array should have the same number of rows as the length of theta_list."
    
    len_theta = len(theta_list)
    len_x = position_array.shape[1]
    
    exp_list = np.exp(-position_array**2 / 2.0)
    norm_list = [np.pi**(-0.25) / np.sqrt(2.0**m * math.factorial(m)) for m in range(dim)]
    herm_list = [np.polyval(hermite(m), position_array) for m in range(dim)]
    
    product_matrix = np.zeros((len_theta, len_x, dim, dim), dtype=np.complex128)
    
    for m in range(dim):
        psi_m = exp_list * norm_list[m] * herm_list[m]
        for n in range(dim):
            psi_n = exp_list * norm_list[n] * herm_list[n]
            phase_factor = np.exp(1j * (m - n) * (theta_list[:, None] - np.pi / 2))
            product_matrix[:, :, m, n] = psi_m * psi_n * phase_factor
    
    return product_matrix
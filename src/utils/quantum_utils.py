import qutip as qt
import numpy as np

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

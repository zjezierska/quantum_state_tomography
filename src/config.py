import numpy as np
import qutip as qt

# Configuration constants
DIMS = 4 # Dimension of the quantum state
EVO_DIMS = 40 # Evolution dimensions
NUM_SAMPLES = 32 # Number of samples to generate
TRAJ_LENGTH = 64 # Number of time steps in the trajectory
NOF_SAMPLES_DISTR = 100 # Number of position samples to sample from the distribution
OUTPUT_DIR = 'data' # Output directory for saving the generated data
X_MAX = 5 # Maximum position value (absolute value)
X_LIST = np.linspace(-X_MAX, X_MAX, NOF_SAMPLES_DISTR) # Position list for psi products
T_LIST = np.linspace(0, 2 * np.pi, TRAJ_LENGTH) # Time list for the trajectory

# Quantum Definitions
ALPHA = 0.1 # Quartic potential parameter
GAMMA = 0 # Decoherence rate

# Create quantum operators using QuTiP
a = qt.destroy(DIMS) # Annihilation operator
x = a.dag() + a # Position operator
p = 1j * (a.dag() - a) # Momentum operator
H_QUARTIC = p**2 / 4 + (x / ALPHA)**4 # Quartic potential Hamiltonian
H_HARMONIC = a.dag() * a # Harmonic oscillator Hamiltonian
c_ops = [np.sqrt(GAMMA) * x]  # Decoherence operators

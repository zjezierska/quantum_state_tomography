import numpy as np
import qutip as qt

# Configuration constants
DIMS = 4 # Dimension of the quantum state
EVO_DIMS = 40 # Evolution dimensions
NUM_SAMPLES = 512 # Number of samples to generate
TRAJ_LENGTH = 64 # Number of time steps in the trajectory
NOF_SAMPLES_DISTR = 100 # Number of position samples to sample from the distribution
OUTPUT_DIR = 'data' # Output directory for saving the generated data
X_MAX = 5 # Maximum position value (absolute value)
BATCH_SIZE_GEN = 32 # Batch size for generating data 

# Quantum Definitions
ALPHA = 0.1 # Quartic potential parameter
GAMMA = 0 # Decoherence rate

# Create quantum operators using QuTiP
a = qt.destroy(DIMS) # Annihilation operator
x = a.dag() + a # Position operator
p = 1j * (a.dag() - a) # Momentum operator
H_QUARTIC = (p*p) / 4 + ((x / ALPHA) * (x / ALPHA) * (x / ALPHA) * (x / ALPHA)) # Quartic potential Hamiltonian
H_HARMONIC = a.dag() * a # Harmonic oscillator Hamiltonian
c_ops = [np.sqrt(GAMMA) * x]  # Decoherence operators
print("Quantum operators created successfully.")

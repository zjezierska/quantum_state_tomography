import unittest
import numpy as np
from src.models.mle import MLEReconstructor

class TestMLEReconstructor(unittest.TestCase):
    
    def setUp(self):
        self.mle_reconstructor = MLEReconstructor('data/raw/states.npy', 'data/raw/trajectories.npy', iterations=10)
        self.mle_reconstructor.states = np.random.rand(10, 4)  # Mocking states data
        self.mle_reconstructor.trajectories = np.random.rand(10, 50, 100)  # Mocking trajectories data

    def test_load_data(self):
        self.mle_reconstructor.load_data()
        self.assertIsNotNone(self.mle_reconstructor.states)
        self.assertIsNotNone(self.mle_reconstructor.trajectories)
        
    def test_mle_algorithm(self):
        theta_data = np.linspace(0, 2 * np.pi, self.mle_reconstructor.trajectories.shape[1])
        sample = self.mle_reconstructor.trajectories[0]
        final_rho, likelihood_trend = self.mle_reconstructor.mle_algorithm(sample, theta_data, dims=2, num_iters=10)

        # Check if trace of rho equals 1
        self.assertTrue(np.isclose(np.trace(final_rho), 1), "The trace of the final density matrix is not equal to 1.")
        
        # Check if rho is Hermitian
        self.assertTrue(np.allclose(final_rho, final_rho.conj().T), "The final density matrix is not Hermitian.")
        
        # Check if rho is positive semi-definite
        eigenvalues = np.linalg.eigvals(final_rho)
        self.assertTrue(np.all(eigenvalues >= 0), f"The final density matrix is not positive semi-definite. Eigenvalues: {eigenvalues}")

        
    def test_calculate_likelihood(self):
        likelihood = self.mle_reconstructor.calculate_likelihood(np.random.rand(4, 4), np.random.rand(10, 50, 100))
        self.assertTrue(0 <= likelihood <= 1)
        
    def test_compare_matrices(self):
        fidelity = self.mle_reconstructor.compare_matrices(np.random.rand(4, 4), np.random.rand(4, 4))
        self.assertTrue(0 <= fidelity <= 1)

if __name__ == '__main__':
    unittest.main()
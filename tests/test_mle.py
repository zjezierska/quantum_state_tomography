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
        self.mle_reconstructor.mle_algorithm()
        self.assertIsNotNone(self.mle_reconstructor.estimated_states)
        
    def test_calculate_likelihood(self):
        likelihood = self.mle_reconstructor.calculate_likelihood(np.random.rand(4, 4), np.random.rand(10, 50, 100))
        self.assertTrue(0 <= likelihood <= 1)
        
    def test_compare_matrices(self):
        fidelity = self.mle_reconstructor.compare_matrices(np.random.rand(4, 4), np.random.rand(4, 4))
        self.assertTrue(0 <= fidelity <= 1)

if __name__ == '__main__':
    unittest.main()
"""
This module calculates MLE (maximum likelihood estimation) for imported quadrature data.

Our states are in the states.npy file and it's a 2D matrix of size n x 2*dim**2 where
n is the number of samples (quantum states) and dim is the dimension of the quantum state.
Our trajectories are in the trajectories.npy file and it's a 3D matrix of size n x TRAJ_LENGTH x NOF_SAMPLES_DISTR
where n is the number of samples (quantum states), TRAJ_LENGTH is the number of time steps in the trajectory
(time_steps = np.linspace(0, 2 * np.pi, TRAJ_LENGTH)) and NOF_SAMPLES_DISTR is the number of position samples to sample from the distribution.

The code will work for 'iters' iterations and will attempt to find the state matrix creating the
trajectories matrix using MLE. The code will print the likelihood of the rho matrix at each step.
The code will then compare the created matrix with the true matrix and print the fidelity between them.

run:
python src/models/mle.py --states /path/to/states.npy --trajectories /path/to/trajectories.npy --iterations 100 --visualize

Copyright (c) 2024 Zuzanna Jezierska
"""

import numpy as np
import qutip as qt
import time
import math
from scipy.special import hermite
import matplotlib.pyplot as plt
from typing import Tuple, List
from src.utils.quantum_utils import my_fidelity


class MLEReconstructor:
    def __init__(self, states_file, trajectories_file, iterations=100,  visualize_steps=False):
        self.states_file = states_file
        self.trajectories_file = trajectories_file
        self.iterations = iterations
        self.visualize_steps = visualize_steps
        self.states = None
        self.trajectories = None

    def load_data(self):
            """
            Loads the states and trajectories from the specified files.

            This method loads the states and trajectories from the files specified by `states_file` and `trajectories_file`
            respectively. The loaded data is stored in the `states` and `trajectories` attributes of the class.

            Args:
                None

            Returns:
                None
            """
            self.states = np.load(self.states_file)
            self.trajectories = np.load(self.trajectories_file)
    
    def calculate_psi_products(position_array: np.ndarray, theta_list: np.ndarray, dim: int) -> np.ndarray:
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

    def mle_algorithm(self, sample: np.ndarray, theta_data: np.ndarray, dims: int, num_iters: int) -> Tuple[np.ndarray, List[float]]:
        """
        Performs maximum likelihood estimation (MLE) algorithm for quantum state reconstruction.

        Args:
            sample (np.ndarray): The quadrature data sample.
            theta_data (np.ndarray): The theta data.
            dims (int): The dimension of the density matrix.
            num_iters (int): The number of iterations to perform.

        Returns:
            Tuple[np.ndarray, List[float]]: A tuple containing the updated density matrix and a list of likelihood values at each iteration.
        """
        
        likelihood_trend = [] # will be used to keep track of the likelihood of rho at each step
        
        n_list = np.linspace(0, dims - 1, dims, dtype=int) # creates an integer list of the values of n (Fock states) we are considering

        rho = np.zeros((dims, dims), complex) # initialize the correct dimension density matrix
        np.fill_diagonal(rho, 1) # fill the diagonals with ones
        rho = rho / np.sum(rho.diagonal()) # normalize the trace of the density matrix
        
        # evaluation of the wavefunction for theta_data, sample.shape = (len(theta_data), NOF_SAMPLES_DISTR), dims
        # the output is of shape (len(theta_data), NOF_SAMPLES_DISTR, dims, dims)
        psi_matrix = self.calculate_psi_products(sample, theta_data, dims) # calculate the psi matrix for the quadrature data
        
        likelihood = np.real(np.sum(np.log(np.sum(np.sum(np.sum(psi_matrix * np.real(rho), axis=2), axis=2), axis=1))))
        likelihood_trend.append(likelihood) # add likelihood to trend list
        
        for i in range(num_iters): # repeat the calculation for num_iters iterations

            # calculate the R matrix
            R_matrix = np.real(np.diag(1 / np.sum(np.sum(np.sum(psi_matrix * np.real(rho), axis=1), axis=0), axis=0)))
            
            # Calculate the new rho using matrix multiplications
            psi_conj = psi_matrix.conj()

            # Contract psi_conj with R_matrix along the last axis of psi_conj and first axis of R_matrix
            temp1 = np.einsum('ijkl,ln->ijkn', psi_conj, R_matrix)
            
            # Contract the result with psi_matrix along the last axis of temp1 and second to last axis of psi_matrix
            temp2 = np.einsum('ijkn,ijmn->kmn', temp1, psi_matrix)

            # Perform the final contraction with rho along the middle dimensions
            rho_update = np.einsum('kmn,nm->kn', temp2, rho)

            # Normalize rho to have trace 1
            rho = rho_update / np.trace(rho_update)
            
            # calculate the likelihood of the updated rho
            likelihood = np.real(np.sum(np.log(np.sum(np.sum(np.sum(psi_matrix * np.real(rho), axis=2), axis=2), axis=1))))
            likelihood_trend.append(likelihood) # add likelihood to trend list
            
            if self.visualize_steps: # visualize how the diagonal elements of the density matrix evolve over time
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(n_list, np.diag(np.real(rho)), color='lightcoral', edgecolor='black', label=f'iteration {i + 1}')
                ax.set_xticks(n_list)
                ax.grid(linestyle='--')
                ax.set_axisbelow(True)
                ax.set_xlabel(r'$n$', fontsize=16)
                ax.set_ylabel(r'$\rho_{nn}$', fontsize=16)
                ax.legend(fontsize=14)
                plt.show()
        
        return rho, likelihood_trend

    def compare_matrices(self, true_matrix, estimated_matrix):
        return my_fidelity(true_matrix, estimated_matrix)

    def run(self):
            """
            Runs the maximum likelihood estimation algorithm on the given trajectories.

            Returns:
                all_likelihood_trends (list): A list of likelihood trends for each trajectory.
            """
            
            self.load_data()
            dims = int(np.sqrt(self.states.shape[1] // 2))
            num_iters = self.iterations
            theta_data = np.linspace(0, 2 * np.pi, self.trajectories.shape[1])

            all_likelihood_trends = []

            for i in range(self.trajectories.shape[0]):
                sample = self.trajectories[i]
                _, likelihood_trend = self.mle_algorithm(sample, theta_data, dims, num_iters)
                all_likelihood_trends.append(likelihood_trend)

            if len(all_likelihood_trends) > 1:
                # Calculate the average likelihood trend
                average_likelihood_trend = np.mean(all_likelihood_trends, axis=0)
                print(f"Average Likelihood Trend: {average_likelihood_trend}")
            else:
                # Print the single likelihood trend
                print(f"Likelihood Trend: {all_likelihood_trends[0]}")

            return all_likelihood_trends


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MLE for quantum state reconstruction.")
    parser.add_argument('--states', type=str, required=True, help='Path to states.npy file')
    parser.add_argument('--trajectories', type=str, required=True, help='Path to trajectories.npy file')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations for MLE')
    parser.add_argument('--visualize', action='store_true', help='Visualize intermediate steps')

    args = parser.parse_args()

    mle_reconstructor = MLEReconstructor(args.states, args.trajectories, args.iterations, args.visualize)
    mle_reconstructor.run()

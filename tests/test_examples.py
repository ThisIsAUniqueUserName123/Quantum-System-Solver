"""
Example tests demonstrating common quantum mechanical systems.
These tests serve as both validation and documentation of the solver's capabilities.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.LinearOperatorSystem import LinearOperatorSystem
from src.BoundaryConditions import (
    zero_boundary_condition,
    periodic_boundary_condition,
    polar_boundary_condition_2D
)
from src.MomentumCoordinateChange import (
    get_kinetic_energy,
    vectorize_coefficients
)

# Mark these as example tests
pytestmark = pytest.mark.integration


class TestClassicalLimits:
    """Test that quantum systems approach classical limits appropriately."""
    
    def test_high_energy_particle_in_box(self):
        """Test that high energy eigenstates show classical behavior."""
        L = 10
        bounds = [(0, L)]
        num_divisions = [200]
        
        operator_dict = {
            (2,): lambda x: -0.5 * np.ones_like(x),
        }
        
        boundary_conditions = [zero_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        
        # Get a high energy state
        eigenvalues, eigenvectors = system.compute_eigenstates(50)
        
        # Find the 50th eigenstate
        idx = np.argsort(np.real(eigenvalues))[49]
        high_state = np.abs(eigenvectors[:, idx])**2
        
        # In the classical limit, the probability should be uniform
        # Check that the variation is small in the bulk
        bulk = high_state[20:-20]  # Exclude boundaries
        relative_variation = np.std(bulk) / np.mean(bulk)
        
        # High quantum number should give nearly uniform distribution
        assert relative_variation < 0.5
        
    def test_coherent_state_dynamics(self):
        """Test that coherent states behave classically in harmonic oscillator."""
        bounds = [(-6, 6)]
        num_divisions = [150]
        
        # Displaced harmonic oscillator
        x0 = 2.0  # Displacement
        operator_dict = {
            (2,): lambda x: -0.5 * np.ones_like(x),
            (0,): lambda x: 0.5 * (x - x0)**2
        }
        
        boundary_conditions = [zero_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        eigenvalues, eigenvectors = system.compute_eigenstates(1)
        
        # Ground state should be centered at x0
        ground_state = np.abs(eigenvectors[:, 0])**2
        x_grid = system.grid_points[0]
        expectation_x = np.sum(x_grid * ground_state) * system.dx[0]
        
        # Check that the wavepacket is centered at the displacement
        assert abs(expectation_x - x0) < 0.1


class TestSymmetries:
    """Test that the solver correctly handles symmetries."""
    
    def test_parity_symmetry(self):
        """Test parity symmetry in symmetric potentials."""
        bounds = [(-5, 5)]
        num_divisions = [101]  # Odd number for symmetry
        
        # Symmetric potential V(x) = x^4 - 2x^2
        operator_dict = {
            (2,): lambda x: -0.5 * np.ones_like(x),
            (0,): lambda x: x**4 - 2*x**2
        }
        
        boundary_conditions = [zero_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        eigenvalues, eigenvectors = system.compute_eigenstates(4)
        
        # Check parity of first few eigenstates
        x_grid = system.grid_points[0]
        mid_point = num_divisions[0] // 2
        
        for i in range(4):
            state = np.real(eigenvectors[:, i])
            # Check if state is even or odd
            left_half = state[:mid_point]
            right_half = state[-mid_point:][::-1]
            
            # State should be either even or odd
            even_corr = np.corrcoef(left_half, right_half)[0, 1]
            odd_corr = np.corrcoef(left_half, -right_half)[0, 1]
            
            assert abs(even_corr) > 0.95 or abs(odd_corr) > 0.95
            
    def test_rotational_symmetry(self):
        """Test rotational symmetry in 2D circular well."""
        # Use polar coordinates
        bounds = [(0.1, 3), (0, 2*np.pi)]
        num_divisions = [30, 60]
        
        # Circular infinite well
        operator_dict = {
            (2, 0): lambda r, theta: -0.5 * np.ones_like(r),
            (1, 0): lambda r, theta: -0.5 / r,
            (0, 2): lambda r, theta: -0.5 / r**2,
            (0, 0): lambda r, theta: np.zeros_like(r)  # V=0 inside
        }
        
        boundary_conditions = [
            zero_boundary_condition(0),
            periodic_boundary_condition(1)
        ]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        eigenvalues, eigenvectors = system.compute_eigenstates(3)
        
        # Ground state should be rotationally symmetric
        ground_state = np.abs(eigenvectors[..., 0])**2
        
        # Check angular independence
        angular_variation = np.std(ground_state, axis=1) / np.mean(ground_state, axis=1)
        mean_variation = np.mean(angular_variation[5:-5])  # Exclude boundaries
        
        assert mean_variation < 0.1


class TestQuantumTunneling:
    """Test quantum tunneling phenomena."""
    
    def test_tunneling_through_barrier(self):
        """Test tunneling through a finite potential barrier."""
        bounds = [(-10, 10)]
        num_divisions = [400]
        
        # Potential barrier
        def barrier_potential(x):
            V = np.zeros_like(x)
            barrier_mask = np.abs(x) < 1
            V[barrier_mask] = 5.0  # Barrier height
            return V
            
        operator_dict = {
            (2,): lambda x: -0.5 * np.ones_like(x),
            (0,): barrier_potential
        }
        
        boundary_conditions = [zero_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        eigenvalues, eigenvectors = system.compute_eigenstates(10)
        
        # Find states with energy less than barrier height
        tunneling_states = np.real(eigenvalues) < 5.0
        
        if np.any(tunneling_states):
            # Check that wavefunction is non-zero on both sides of barrier
            idx = np.where(tunneling_states)[0][0]
            state = np.abs(eigenvectors[:, idx])**2
            
            left_region = state[:150]  # x < -1
            right_region = state[-150:]  # x > 1
            
            # Should have non-negligible probability on both sides
            assert np.max(left_region) > 1e-6
            assert np.max(right_region) > 1e-6
            
    def test_double_well_splitting(self):
        """Test energy splitting in double-well potential."""
        bounds = [(-4, 4)]
        num_divisions = [200]
        
        # Double well potential: V(x) = (x^2 - 1)^2
        operator_dict = {
            (2,): lambda x: -0.5 * np.ones_like(x),
            (0,): lambda x: (x**2 - 1)**2
        }
        
        boundary_conditions = [zero_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        eigenvalues, eigenvectors = system.compute_eigenstates(4)
        
        # Sort eigenvalues
        sorted_E = np.sort(np.real(eigenvalues))
        
        # First two states should be nearly degenerate (tunnel splitting)
        splitting = sorted_E[1] - sorted_E[0]
        
        # Splitting should be small but non-zero
        assert 0 < splitting < 0.1


class TestNumericalConvergence:
    """Test numerical convergence properties."""
    
    @pytest.mark.slow
    def test_grid_convergence(self):
        """Test that results converge with grid refinement."""
        bounds = [(-5, 5)]
        
        # Harmonic oscillator
        operator_dict_func = lambda: {
            (2,): lambda x: -0.5 * np.ones_like(x),
            (0,): lambda x: 0.5 * x**2
        }
        
        ground_energies = []
        grid_sizes = [50, 100, 200]
        
        for n in grid_sizes:
            system = LinearOperatorSystem(
                bounds, [n], operator_dict_func(),
                [zero_boundary_condition(0)]
            )
            eigenvalues, _ = system.compute_eigenstates(1)
            ground_energies.append(np.min(np.real(eigenvalues)))
            
        # Check convergence
        errors = [abs(E - 0.5) for E in ground_energies]
        
        # Errors should decrease
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]
        
        # Should converge to analytical value
        assert abs(ground_energies[-1] - 0.5) < 0.01
        
    def test_accuracy_order_convergence(self):
        """Test that higher accuracy orders converge faster."""
        bounds = [(-5, 5)]
        num_divisions = [60]
        
        operator_dict = {
            (2,): lambda x: -0.5 * np.ones_like(x),
            (0,): lambda x: 0.5 * x**2
        }
        
        boundary_conditions = [zero_boundary_condition(0)]
        
        errors = []
        orders = [2, 4, 6]
        
        for order in orders:
            system = LinearOperatorSystem(
                bounds, num_divisions, operator_dict,
                boundary_conditions, accuracy_order=order
            )
            eigenvalues, _ = system.compute_eigenstates(1)
            ground_energy = np.min(np.real(eigenvalues))
            errors.append(abs(ground_energy - 0.5))
            
        # Higher order should give better accuracy
        assert errors[1] < errors[0] * 0.1  # 4th order much better than 2nd
        assert errors[2] < errors[1] * 0.1  # 6th order much better than 4th


@pytest.mark.requires_display
class TestVisualization:
    """Test visualization capabilities."""
    
    def test_1D_eigenstate_plotting(self):
        """Test that 1D eigenstates can be plotted."""
        bounds = [(-5, 5)]
        num_divisions = [100]
        
        operator_dict = {
            (2,): lambda x: -0.5 * np.ones_like(x),
            (0,): lambda x: 0.5 * x**2
        }
        
        boundary_conditions = [zero_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        eigenvalues, eigenvectors = system.compute_eigenstates(3)
        
        # This should not raise an error
        try:
            # Suppress display for testing
            plt.ion()
            system.plot_eigenstates(eigenvectors, 3)
            plt.close('all')
        except Exception as e:
            pytest.fail(f"Plotting failed: {e}")
            
    def test_2D_eigenstate_plotting(self):
        """Test that 2D eigenstates can be plotted."""
        bounds = [(-3, 3), (-3, 3)]
        num_divisions = [30, 30]
        
        operator_dict = {
            (2, 0): lambda x, y: -0.5 * np.ones_like(x),
            (0, 2): lambda x, y: -0.5 * np.ones_like(x),
            (0, 0): lambda x, y: 0.5 * (x**2 + y**2)
        }
        
        boundary_conditions = [zero_boundary_condition(0), zero_boundary_condition(1)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        eigenvalues, eigenvectors = system.compute_eigenstates(1)
        
        try:
            plt.ion()
            system.plot_eigenstates(eigenvectors, 1)
            plt.close('all')
        except Exception as e:
            pytest.fail(f"2D plotting failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

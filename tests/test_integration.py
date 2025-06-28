"""
Integration tests for the Quantum System Solver.
Tests the complete workflow from problem setup to eigenstate computation.
"""

import numpy as np
import pytest
import sympy as sp
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


class TestQuantumSystems:
    """Test complete quantum system solutions."""
    
    def test_1D_particle_in_box(self):
        """Test 1D particle in a box (infinite square well)."""
        # System parameters
        L = 10  # Box length
        bounds = [(0, L)]
        num_divisions = [100]
        
        # Only kinetic energy (V = 0 inside the box)
        operator_dict = {
            (2,): lambda x: -0.5 * np.ones_like(x)  # -ħ²/(2m) d²/dx² with ħ=m=1
        }
        
        # Zero boundary conditions (wave function vanishes at walls)
        boundary_conditions = [zero_boundary_condition(0)]
        
        # Create and solve the system
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        eigenvalues, eigenvectors = system.compute_eigenstates(5)
        
        # Analytical solutions: E_n = n²π²/(2L²) for n = 1, 2, 3, ...
        expected_energies = np.array([n**2 * np.pi**2 / (2 * L**2) for n in range(1, 6)])
        
        # Sort computed eigenvalues
        sorted_eigenvalues = np.sort(np.real(eigenvalues))
        
        # Check eigenvalues (allowing for numerical error)
        np.testing.assert_allclose(sorted_eigenvalues, expected_energies, rtol=0.01)
        
    def test_1D_harmonic_oscillator(self):
        """Test 1D quantum harmonic oscillator."""
        # System parameters  
        bounds = [(-6, 6)]
        num_divisions = [150]
        
        # Hamiltonian: -1/2 d²/dx² + 1/2 x²
        operator_dict = {
            (2,): lambda x: -0.5 * np.ones_like(x),  # Kinetic energy
            (0,): lambda x: 0.5 * x**2               # Potential energy
        }
        
        boundary_conditions = [zero_boundary_condition(0)]
        
        # Create and solve the system
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        eigenvalues, eigenvectors = system.compute_eigenstates(6)
        
        # Analytical solutions: E_n = n + 1/2 for n = 0, 1, 2, ...
        expected_energies = np.array([n + 0.5 for n in range(6)])
        
        # Sort computed eigenvalues
        sorted_eigenvalues = np.sort(np.real(eigenvalues))
        
        # Check eigenvalues
        np.testing.assert_allclose(sorted_eigenvalues, expected_energies, atol=0.05)
        
        # Check ground state shape (should be Gaussian-like)
        ground_state_idx = np.argmin(np.real(eigenvalues))
        ground_state = np.real(eigenvectors[:, ground_state_idx])
        
        # Ground state should have no nodes
        sign_changes = np.sum(np.diff(np.sign(ground_state)) != 0)
        assert sign_changes <= 2  # Allow for numerical noise near zero
        
    def test_2D_harmonic_oscillator(self):
        """Test 2D quantum harmonic oscillator."""
        bounds = [(-4, 4), (-4, 4)]
        num_divisions = [40, 40]
        
        # 2D Harmonic oscillator: H = -1/2(∂²/∂x² + ∂²/∂y²) + 1/2(x² + y²)
        operator_dict = {
            (2, 0): lambda x, y: -0.5 * np.ones_like(x),
            (0, 2): lambda x, y: -0.5 * np.ones_like(x),
            (0, 0): lambda x, y: 0.5 * (x**2 + y**2)
        }
        
        boundary_conditions = [zero_boundary_condition(0), zero_boundary_condition(1)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        eigenvalues, eigenvectors = system.compute_eigenstates(3)
        
        # Ground state energy should be 1.0 (nx=0, ny=0: E = 0.5 + 0.5)
        ground_energy = np.min(np.real(eigenvalues))
        assert abs(ground_energy - 1.0) < 0.1
        
        # First excited states should have energy 2.0
        sorted_eigenvalues = np.sort(np.real(eigenvalues))
        assert abs(sorted_eigenvalues[1] - 2.0) < 0.2
        
    def test_periodic_system(self):
        """Test system with periodic boundary conditions."""
        # System on a ring
        bounds = [(0, 2*np.pi)]
        num_divisions = [100]
        
        # Free particle on a ring
        operator_dict = {
            (2,): lambda x: -0.5 * np.ones_like(x)
        }
        
        boundary_conditions = [periodic_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        eigenvalues, eigenvectors = system.compute_eigenstates(5)
        
        # Analytical solutions: E_n = n²/2 for n = 0, ±1, ±2, ...
        # Lowest energies: 0, 0.5, 0.5, 2, 2, 4.5, 4.5, ...
        expected_energies = np.array([0, 0.5, 0.5, 2.0, 2.0])
        
        sorted_eigenvalues = np.sort(np.real(eigenvalues))
        
        # Check ground state
        assert abs(sorted_eigenvalues[0]) < 0.01
        
        # Check first excited state (doubly degenerate)
        assert abs(sorted_eigenvalues[1] - 0.5) < 0.05
        assert abs(sorted_eigenvalues[2] - 0.5) < 0.05
        
    def test_polar_coordinates_system(self):
        """Test system using polar coordinate transformation."""
        # First, get the kinetic energy operator in polar coordinates
        def to_cartesian(r, theta):
            return r * sp.cos(theta), r * sp.sin(theta)
            
        derivatives, coords = get_kinetic_energy(to_cartesian, m=1)
        
        # Add a radial harmonic potential
        r = coords[0]
        derivatives[(0, 0)] = derivatives.get((0, 0), 0) + r**2 / 2
        
        # Vectorize the coefficients
        vectorized = vectorize_coefficients(derivatives, coords)
        
        # Set up the system
        bounds = [(0.1, 5), (0, 2*np.pi)]  # r, theta
        num_divisions = [50, 30]
        
        # Use the vectorized coefficients
        operator_dict = vectorized
        
        # Polar boundary conditions
        boundary_conditions = [
            zero_boundary_condition(0),      # Zero at r boundary
            periodic_boundary_condition(1)    # Periodic in theta
        ]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        eigenvalues, eigenvectors = system.compute_eigenstates(3)
        
        # Check that we get real eigenvalues
        assert np.all(np.abs(np.imag(eigenvalues)) < 1e-10)
        
        # Check that eigenstates are normalized
        for i in range(3):
            norm = np.sum(np.abs(eigenvectors[..., i])**2)
            assert abs(norm - 1.0) < 0.01
            
    def test_different_accuracy_orders(self):
        """Test that higher accuracy orders give better results."""
        bounds = [(-5, 5)]
        num_divisions = [50]
        
        # 1D harmonic oscillator
        operator_dict = {
            (2,): lambda x: -0.5 * np.ones_like(x),
            (0,): lambda x: 0.5 * x**2
        }
        
        boundary_conditions = [zero_boundary_condition(0)]
        
        # Compute with different accuracy orders
        errors = []
        for accuracy in [2, 4, 6]:
            system = LinearOperatorSystem(bounds, num_divisions, operator_dict, 
                                         boundary_conditions, accuracy_order=accuracy)
            eigenvalues, _ = system.compute_eigenstates(1)
            
            # Ground state energy should be 0.5
            ground_energy = np.min(np.real(eigenvalues))
            errors.append(abs(ground_energy - 0.5))
            
        # Higher accuracy should give smaller errors
        assert errors[1] < errors[0]  # 4th order better than 2nd
        assert errors[2] < errors[1]  # 6th order better than 4th
        
    def test_complex_potential(self):
        """Test system with complex potential (non-Hermitian Hamiltonian)."""
        bounds = [(-3, 3)]
        num_divisions = [80]
        
        # Complex potential: V(x) = x² + i*x
        operator_dict = {
            (2,): lambda x: -0.5 * np.ones_like(x),
            (0,): lambda x: x**2 + 1j * x
        }
        
        boundary_conditions = [zero_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        eigenvalues, eigenvectors = system.compute_eigenstates(3)
        
        # With complex potential, eigenvalues should be complex
        assert np.any(np.abs(np.imag(eigenvalues)) > 0.01)
        
        # Check normalization still works
        for i in range(3):
            norm = np.sum(np.abs(eigenvectors[:, i])**2)
            assert abs(norm - 1.0) < 0.01


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_mismatched_dimensions(self):
        """Test that mismatched dimensions raise appropriate errors."""
        bounds = [(-1, 1), (-1, 1)]  # 2D
        num_divisions = [10, 10]
        
        # Wrong: 1D derivative specification for 2D system
        operator_dict = {(2,): lambda x, y: np.ones_like(x)}
        
        boundary_conditions = [zero_boundary_condition(0), zero_boundary_condition(1)]
        
        with pytest.raises(ValueError):
            LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
            
    def test_invalid_eigenstate_request(self):
        """Test requesting more eigenstates than system size."""
        bounds = [(-1, 1)]
        num_divisions = [5]  # Very small system
        
        operator_dict = {(2,): lambda x: -0.5 * np.ones_like(x)}
        boundary_conditions = [zero_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        
        # Try to compute more eigenstates than system dimension
        with pytest.raises(Exception):  # scipy will raise an error
            eigenvalues, eigenvectors = system.compute_eigenstates(10)
            
    def test_empty_operator(self):
        """Test system with empty operator dictionary."""
        bounds = [(-1, 1)]
        num_divisions = [10]
        
        operator_dict = {}  # Empty operator
        boundary_conditions = [zero_boundary_condition(0)]
        
        # Should create system but operator will be zero
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        
        # All eigenvalues should be zero
        eigenvalues, eigenvectors = system.compute_eigenstates(3)
        np.testing.assert_array_almost_equal(eigenvalues, np.zeros(3))


class TestPerformance:
    """Test performance-related aspects."""
    
    def test_sparse_matrix_efficiency(self):
        """Test that operator matrix is efficiently sparse."""
        bounds = [(-5, 5), (-5, 5)]
        num_divisions = [30, 30]
        
        operator_dict = {
            (2, 0): lambda x, y: -0.5 * np.ones_like(x),
            (0, 2): lambda x, y: -0.5 * np.ones_like(x)
        }
        
        boundary_conditions = [zero_boundary_condition(0), zero_boundary_condition(1)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        
        # Check sparsity
        total_elements = system.N * system.N
        nonzero_elements = system.operator.nnz
        sparsity = nonzero_elements / total_elements
        
        # For finite difference operators, sparsity should be very low
        assert sparsity < 0.01  # Less than 1% non-zero elements
        
    def test_eigenstate_computation_scales(self):
        """Test that eigenstate computation works for different system sizes."""
        sizes = [10, 20, 40]
        
        for n in sizes:
            bounds = [(-2, 2)]
            num_divisions = [n]
            
            operator_dict = {
                (2,): lambda x: -0.5 * np.ones_like(x),
                (0,): lambda x: 0.5 * x**2
            }
            
            boundary_conditions = [zero_boundary_condition(0)]
            
            system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
            eigenvalues, eigenvectors = system.compute_eigenstates(2)
            
            # Should successfully compute for all sizes
            assert len(eigenvalues) == 2
            assert eigenvectors.shape == (n, 2)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])

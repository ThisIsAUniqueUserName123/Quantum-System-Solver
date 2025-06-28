"""
Test suite for LinearOperatorSystem class.
Tests the core functionality of the quantum system solver.
"""

import numpy as np
import pytest
from scipy.sparse import issparse
import sys
import os

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.LinearOperatorSystem import LinearOperatorSystem
from src.BoundaryConditions import (
    zero_boundary_condition, 
    periodic_boundary_condition,
    polar_boundary_condition_2D,
    polar_boundary_condition_3D,
    radial_boundary_condition
)

class TestLinearOperatorSystem:
    """Test the LinearOperatorSystem class functionality."""
    
    def test_initialization_1D(self):
        """Test initialization of a 1D system."""
        bounds = [(-5, 5)]
        num_divisions = [100]
        operator_dict = {(2,): lambda x: -0.5*np.ones_like(x)}  # -1/2 * d^2/dx^2
        boundary_conditions = [zero_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        
        assert system.num_dims == 1
        assert system.N == 100
        assert len(system.grid_points) == 1
        assert system.grid_points[0].shape == (100,)
        assert np.isclose(system.dx[0], 0.1010101, rtol=1e-5)
        
    def test_initialization_2D(self):
        """Test initialization of a 2D system."""
        bounds = [(-5, 5), (-3, 3)]
        num_divisions = [50, 30]
        operator_dict = {
            (2, 0): lambda x, y: -0.5*np.ones_like(x),  # -1/2 * d^2/dx^2
            (0, 2): lambda x, y: -0.5*np.ones_like(x)   # -1/2 * d^2/dy^2
        }
        boundary_conditions = [zero_boundary_condition(0), zero_boundary_condition(1)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        
        assert system.num_dims == 2
        assert system.N == 1500
        assert len(system.grid_points) == 2
        assert system.grid_points[0].shape == (50,)
        assert system.grid_points[1].shape == (30,)
        assert system.grid.shape == (2, 50, 30)
        
    def test_get_kernel_1D_first_derivative(self):
        """Test 1D kernel generation for first derivative."""
        bounds = [(-1, 1)]
        num_divisions = [10]
        operator_dict = {(1,): lambda x: np.ones_like(x)}
        boundary_conditions = [zero_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions, accuracy_order=2)
        
        # Test first derivative with 2nd order accuracy
        kernel, center = system.get_kernel_1D(1, accuracy_order=2)
        
        # For first derivative with 2nd order accuracy, we expect [-1, 0, 1] / (2*dx)
        expected_kernel = np.array([-0.5, 0, 0.5])
        np.testing.assert_array_almost_equal(kernel, expected_kernel)
        assert center == 1
        
    def test_get_kernel_1D_second_derivative(self):
        """Test 1D kernel generation for second derivative."""
        bounds = [(-1, 1)]
        num_divisions = [10]
        operator_dict = {(2,): lambda x: np.ones_like(x)}
        boundary_conditions = [zero_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions, accuracy_order=2)
        
        # Test second derivative with 2nd order accuracy  
        kernel, center = system.get_kernel_1D(2, accuracy_order=2)
        
        # For second derivative with 2nd order accuracy, we expect [1, -2, 1] / dx^2
        expected_kernel = np.array([1, -2, 1])
        np.testing.assert_array_almost_equal(kernel, expected_kernel)
        assert center == 1
        
    def test_invalid_accuracy_order(self):
        """Test that invalid accuracy order raises ValueError."""
        bounds = [(-1, 1)]
        num_divisions = [10]
        operator_dict = {(2,): lambda x: np.ones_like(x)}
        boundary_conditions = [zero_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        
        # Try to get a second derivative with first order accuracy (should fail)
        with pytest.raises(ValueError, match="Cannot construct stencil of order 2 with accuracy = 1"):
            system.get_kernel_1D(2, accuracy_order=1)
            
    def test_operator_matrix_is_sparse(self):
        """Test that the operator matrix is sparse."""
        bounds = [(-1, 1)]
        num_divisions = [50]
        operator_dict = {(2,): lambda x: -0.5*np.ones_like(x)}
        boundary_conditions = [zero_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        
        assert issparse(system.operator)
        assert system.operator.shape == (50, 50)
        
    def test_compute_eigenstates_1D_harmonic_oscillator(self):
        """Test eigenstate computation for 1D harmonic oscillator."""
        bounds = [(-5, 5)]
        num_divisions = [100]
        # Hamiltonian: -1/2 * d^2/dx^2 + 1/2 * x^2
        operator_dict = {
            (2,): lambda x: -0.5*np.ones_like(x),  # Kinetic energy
            (0,): lambda x: 0.5*x**2                # Potential energy
        }
        boundary_conditions = [zero_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        
        # Compute first 5 eigenstates
        eigenvalues, eigenvectors = system.compute_eigenstates(5)
        
        # For harmonic oscillator, eigenvalues should be n + 1/2 for n = 0, 1, 2, ...
        expected_eigenvalues = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        
        # Sort eigenvalues (they might not be in order)
        sorted_eigenvalues = np.sort(np.real(eigenvalues))
        
        # Check that eigenvalues are approximately correct
        np.testing.assert_allclose(sorted_eigenvalues, expected_eigenvalues, atol=0.1)
        
        # Check eigenvector shape
        assert eigenvectors.shape == (100, 5)
        
        # Check normalization
        for i in range(5):
            norm = np.sum(np.abs(eigenvectors[:, i])**2)
            assert np.isclose(norm, 1.0, rtol=1e-5)
            
    def test_periodic_boundary_condition(self):
        """Test periodic boundary conditions."""
        bounds = [(-np.pi, np.pi)]
        num_divisions = [100]
        operator_dict = {(2,): lambda x: -0.5*np.ones_like(x)}
        boundary_conditions = [periodic_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        
        # The operator should be created without errors
        assert system.operator is not None
        assert system.operator.shape == (100, 100)
        
    def test_2D_system_eigenstates(self):
        """Test 2D system eigenstate computation."""
        bounds = [(-2, 2), (-2, 2)]
        num_divisions = [30, 30]
        # 2D harmonic oscillator
        operator_dict = {
            (2, 0): lambda x, y: -0.5*np.ones_like(x),  # -1/2 * d^2/dx^2
            (0, 2): lambda x, y: -0.5*np.ones_like(x),  # -1/2 * d^2/dy^2
            (0, 0): lambda x, y: 0.5*(x**2 + y**2)       # 1/2 * (x^2 + y^2)
        }
        boundary_conditions = [zero_boundary_condition(0), zero_boundary_condition(1)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        
        # Compute first 3 eigenstates
        eigenvalues, eigenvectors = system.compute_eigenstates(3)
        
        # For 2D harmonic oscillator, ground state energy is 1.0
        sorted_eigenvalues = np.sort(np.real(eigenvalues))
        assert sorted_eigenvalues[0] > 0.9 and sorted_eigenvalues[0] < 1.1
        
        # Check eigenvector shape
        assert eigenvectors.shape == (30, 30, 3)
        
    def test_invalid_derivative_order_dimension(self):
        """Test that mismatched derivative order dimensions raise an error."""
        bounds = [(-1, 1), (-1, 1)]
        num_divisions = [10, 10]
        # Wrong: using 1D derivative order for 2D system
        operator_dict = {(2,): lambda x, y: np.ones_like(x)}
        boundary_conditions = [zero_boundary_condition(0), zero_boundary_condition(1)]
        
        with pytest.raises(ValueError, match="Derivative order must match"):
            LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
            
    def test_grid_construction(self):
        """Test that grid points are correctly constructed."""
        bounds = [(-1, 1), (0, 2)]
        num_divisions = [3, 4]
        operator_dict = {(0, 0): lambda x, y: np.ones_like(x)}
        boundary_conditions = [zero_boundary_condition(0), zero_boundary_condition(1)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        
        # Check x grid points
        expected_x = np.array([-1, 0, 1])
        np.testing.assert_array_almost_equal(system.grid_points[0], expected_x)
        
        # Check y grid points  
        expected_y = np.linspace(0, 2, 4)
        np.testing.assert_array_almost_equal(system.grid_points[1], expected_y)
        
        # Check meshgrid shape
        assert system.grid.shape == (2, 3, 4)
        
    def test_higher_accuracy_order(self):
        """Test system with higher accuracy order."""
        bounds = [(-1, 1)]
        num_divisions = [50]
        operator_dict = {(2,): lambda x: -0.5*np.ones_like(x)}
        boundary_conditions = [zero_boundary_condition(0)]
        
        # Test with 4th order accuracy
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, 
                                     boundary_conditions, accuracy_order=4)
        
        kernel, center = system.get_kernel_1D(2, accuracy_order=4)
        
        # For 4th order accuracy, the kernel should have 5 points
        assert len(kernel) == 5
        assert center == 2

    def test_complex_coefficients(self):
        """Test that complex coefficients are handled correctly."""
        bounds = [(-1, 1)]
        num_divisions = [50]
        # Complex coefficient
        operator_dict = {
            (1,): lambda x: 1j * np.ones_like(x),  # Imaginary first derivative
            (0,): lambda x: x**2                    # Real potential
        }
        boundary_conditions = [zero_boundary_condition(0)]
        
        system = LinearOperatorSystem(bounds, num_divisions, operator_dict, boundary_conditions)
        
        # Check that operator is complex
        assert system.operator.dtype == np.complex128
        
        # Compute eigenstates should work with complex operators
        eigenvalues, eigenvectors = system.compute_eigenstates(3)
        assert eigenvalues.dtype == np.complex128


class TestBoundaryConditions:
    """Test the boundary condition functions."""
    
    def test_zero_boundary_condition(self):
        """Test zero boundary condition."""
        bc = zero_boundary_condition(0)
        
        # Create test indices
        indices_grid = [np.array([[-1, 0, 1, 2, 3, 4, 5]])]
        num_divisions = [5]
        
        result_indices, mask = bc(indices_grid, num_divisions)
        
        # Check that mask excludes out-of-bounds indices
        expected_mask = np.array([[False, True, True, True, True, True, False]])
        np.testing.assert_array_equal(mask, expected_mask)
        
        # Indices should not be modified
        np.testing.assert_array_equal(result_indices[0], indices_grid[0])
        
    def test_periodic_boundary_condition(self):
        """Test periodic boundary condition."""
        bc = periodic_boundary_condition(0)
        
        # Create test indices with some out-of-bounds
        indices_grid = [np.array([[-1, 0, 1, 4, 5, 6]])]
        num_divisions = [5]
        
        result_indices, mask = bc(indices_grid, num_divisions)
        
        # Check that indices are wrapped
        expected_indices = np.array([[4, 0, 1, 4, 0, 1]])
        np.testing.assert_array_equal(result_indices[0], expected_indices)
        
        # Mask should be None for periodic BC
        assert mask is None
        
    def test_polar_boundary_condition_2D(self):
        """Test 2D polar boundary condition."""
        bc = polar_boundary_condition_2D(radial_axis=0, polar_axis=1)
        
        # Test with negative radial index
        indices_grid = np.meshgrid(np.array([[-1, 0, 1]]), np.array([[2, 3, 4]]), indexing='ij')
        num_divisions = [10, 8]
        
        result_indices, mask = bc(indices_grid, num_divisions)
        
        # First element should have flipped radial coordinate
        np.testing.assert_array_equal(result_indices[0][:, 0], [0,0,1]) # -(-1) - 1 = 0
        
        # First element should have shifted angular coordinate
        np.testing.assert_array_equal(result_indices[1][0, :], [6,7,0]) # (2 + 8//2) % 8 = 6


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])

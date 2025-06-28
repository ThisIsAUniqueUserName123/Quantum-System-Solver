"""
Test suite for MomentumCoordinateChange module.
Tests coordinate transformations and kinetic energy calculations.
"""

import numpy as np
import pytest
import sympy as sp
import sys
import os

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.MomentumCoordinateChange import (
    extract_derivatives,
    get_kinetic_energy,
    get_kinetic_energy_bundle,
    vectorize_coefficients,
    metric_from_jacobian
)


class TestExtractDerivatives:
    """Test the extract_derivatives function."""
    
    def test_simple_derivative(self):
        """Test extraction of a simple derivative."""
        x = sp.Symbol('x')
        psi = sp.Function('psi')(x)
        
        # Create a simple derivative expression
        expr = sp.Derivative(psi, x, 2)
        
        derivatives = extract_derivatives(expr, [x], psi)
        
        # Should have one second-order derivative
        assert (2,) in derivatives
        assert derivatives[(2,)] == 1
        
    def test_derivative_with_coefficient(self):
        """Test extraction of derivative with coefficient."""
        x = sp.Symbol('x')
        psi = sp.Function('psi')(x)
        
        # Create derivative with coefficient
        expr = -0.5 * sp.Derivative(psi, x, 2)
        
        derivatives = extract_derivatives(expr, [x], psi)
        
        assert (2,) in derivatives
        assert derivatives[(2,)] == -0.5
        
    def test_multiple_derivatives(self):
        """Test extraction of multiple derivative terms."""
        x, y = sp.symbols('x y')
        psi = sp.Function('psi')(x, y)
        
        # Create expression with multiple derivatives
        expr = -0.5 * sp.Derivative(psi, x, 2) - 0.5 * sp.Derivative(psi, y, 2)
        
        derivatives = extract_derivatives(expr, [x, y], psi)
        
        assert (2, 0) in derivatives
        assert (0, 2) in derivatives
        assert derivatives[(2, 0)] == -0.5
        assert derivatives[(0, 2)] == -0.5
        
    def test_mixed_derivative(self):
        """Test extraction of mixed derivatives."""
        x, y = sp.symbols('x y')
        psi = sp.Function('psi')(x, y)
        
        # Create mixed derivative  
        expr = 2 * sp.Derivative(psi, x, y)
        
        derivatives = extract_derivatives(expr, [x, y], psi)
        
        assert (1, 1) in derivatives
        assert derivatives[(1, 1)] == 2
        
    def test_constant_term(self):
        """Test extraction of constant (non-derivative) terms."""
        x = sp.Symbol('x')
        psi = sp.Function('psi')(x)
        
        # Expression with constant term
        expr = x**2 * psi
        
        derivatives = extract_derivatives(expr, [x], psi)
        
        # Constant term corresponds to (0,) derivative order
        assert (0,) in derivatives
        assert derivatives[(0,)] == x**2
        
    def test_combined_expression(self):
        """Test complex expression with derivatives and constants."""
        x = sp.Symbol('x')
        psi = sp.Function('psi')(x)
        
        # Hamiltonian-like expression
        expr = -0.5 * sp.Derivative(psi, x, 2) + 0.5 * x**2 * psi
        
        derivatives = extract_derivatives(expr, [x], psi)
        
        assert (2,) in derivatives
        assert (0,) in derivatives  
        assert derivatives[(2,)] == -0.5
        assert derivatives[(0,)] == 0.5 * x**2


class TestKineticEnergy:
    """Test the kinetic energy calculation functions."""
    
    def test_1D_cartesian(self):
        """Test kinetic energy in 1D Cartesian coordinates."""
        # Identity transformation (Cartesian)
        def to_cartesian(x):
            return (x,)
            
        derivatives, coords = get_kinetic_energy(to_cartesian, m=1)
        
        # In Cartesian coordinates, kinetic energy is -1/(2m) * d^2/dx^2
        assert (2,) in derivatives
        # The coefficient should be -1/2 for m=1
        expected_coeff = sp.Rational(-1, 2)
        assert sp.simplify(derivatives[(2,)] - expected_coeff) == 0
        
    def test_2D_cartesian(self):
        """Test kinetic energy in 2D Cartesian coordinates."""
        # Identity transformation (Cartesian)
        def to_cartesian(x, y):
            return x, y
            
        derivatives, coords = get_kinetic_energy(to_cartesian, m=1)
        
        # Should have second derivatives in both x and y
        assert (2, 0) in derivatives
        assert (0, 2) in derivatives
        
        # Both should have coefficient -1/2
        expected_coeff = sp.Rational(-1, 2)
        assert sp.simplify(derivatives[(2, 0)] - expected_coeff) == 0
        assert sp.simplify(derivatives[(0, 2)] - expected_coeff) == 0
        
    def test_polar_coordinates(self):
        """Test kinetic energy in polar coordinates."""
        # Polar to Cartesian transformation
        def to_cartesian(r, theta):
            return r * sp.cos(theta), r * sp.sin(theta)
            
        derivatives, coords = get_kinetic_energy(to_cartesian, m=1)
        
        # In polar coordinates, we expect terms like:
        # -1/(2m) * [d^2/dr^2 + 1/r * d/dr + 1/r^2 * d^2/dθ^2]
        
        r, theta = coords
        
        # Check that we have the right derivative orders
        assert (2, 0) in derivatives  # d^2/dr^2
        assert (0, 2) in derivatives  # d^2/dθ^2
        
    def test_spherical_coordinates(self):
        """Test kinetic energy in spherical coordinates."""
        # Spherical to Cartesian transformation
        def to_cartesian(r, theta, phi):
            return (r * sp.sin(theta) * sp.cos(phi),
                   r * sp.sin(theta) * sp.sin(phi), 
                   r * sp.cos(theta))
                   
        derivatives, coords = get_kinetic_energy(to_cartesian, m=2)
        
        r, theta, phi = coords
        
        # Should have derivatives in all three coordinates
        assert (2, 0, 0) in derivatives  # d^2/dr^2
        assert (0, 2, 0) in derivatives  # d^2/dθ^2
        assert (0, 0, 2) in derivatives  # d^2/dφ^2
        
    def test_kinetic_energy_bundle(self):
        """Test the alternative kinetic energy function."""
        # Test with simple 2D Cartesian
        def to_cartesian(x, y):
            return x, y
            
        derivatives, coords = get_kinetic_energy_bundle(to_cartesian, m=1)
        
        # Should give same result as regular get_kinetic_energy for Cartesian
        assert (2, 0) in derivatives
        assert (0, 2) in derivatives
        
        # Both should have coefficient -1/2
        expected_coeff = sp.Rational(-1, 2)
        assert sp.simplify(derivatives[(2, 0)] - expected_coeff) == 0
        assert sp.simplify(derivatives[(0, 2)] - expected_coeff) == 0


class TestVectorizeCoefficients:
    """Test the vectorize_coefficients function."""
    
    def test_simple_vectorization(self):
        """Test vectorization of simple symbolic expressions."""
        x = sp.Symbol('x')
        
        derivatives = {
            (2,): -sp.Rational(1, 2),
            (0,): x**2 / 2
        }
        
        vectorized = vectorize_coefficients(derivatives, [x])
        
        # Test constant coefficient
        const_func = vectorized[(2,)]
        test_values = np.array([1.0, 2.0, 3.0])
        expected = np.array([-0.5, -0.5, -0.5])
        np.testing.assert_array_almost_equal(const_func(test_values), expected)
        
        # Test x-dependent coefficient
        x_func = vectorized[(0,)]
        expected = test_values**2 / 2
        np.testing.assert_array_almost_equal(x_func(test_values), expected)
        
    def test_2D_vectorization(self):
        """Test vectorization with multiple variables."""
        x, y = sp.symbols('x y')
        
        derivatives = {
            (2, 0): -sp.Rational(1, 2),
            (0, 2): -sp.Rational(1, 2),
            (0, 0): (x**2 + y**2) / 2
        }
        
        vectorized = vectorize_coefficients(derivatives, [x, y])
        
        # Test with 2D arrays
        x_vals = np.array([[1, 2], [3, 4]])
        y_vals = np.array([[1, 1], [2, 2]])
        
        # Test constant coefficients
        const_func = vectorized[(2, 0)]
        expected = np.array([[-0.5, -0.5], [-0.5, -0.5]])
        np.testing.assert_array_almost_equal(const_func(x_vals, y_vals), expected)
        
        # Test position-dependent coefficient
        pot_func = vectorized[(0, 0)]
        expected = (x_vals**2 + y_vals**2) / 2
        np.testing.assert_array_almost_equal(pot_func(x_vals, y_vals), expected)
        
    def test_complex_expression(self):
        """Test vectorization of complex symbolic expressions."""
        r, theta = sp.symbols('r theta', real=True)
        
        derivatives = {
            (2, 0): -sp.Rational(1, 2),
            (1, 0): -1 / (2 * r),
            (0, 2): -1 / (2 * r**2)
        }
        
        vectorized = vectorize_coefficients(derivatives, [r, theta])
        
        # Test with arrays
        r_vals = np.array([1.0, 2.0, 3.0])
        theta_vals = np.array([0, np.pi/2, np.pi])
        
        # Test 1/r coefficient
        r_inv_func = vectorized[(1, 0)]
        expected = -1 / (2 * r_vals)
        np.testing.assert_array_almost_equal(r_inv_func(r_vals, theta_vals), expected)
        
        # Test 1/r^2 coefficient  
        r2_inv_func = vectorized[(0, 2)]
        expected = -1 / (2 * r_vals**2)
        np.testing.assert_array_almost_equal(r2_inv_func(r_vals, theta_vals), expected)


class TestMetricFromJacobian:
    """Test the metric_from_jacobian function."""
    
    def test_2D_cartesian_metric(self):
        """Test metric for 2D Cartesian coordinates."""
        # Identity Jacobian for Cartesian coordinates
        J = sp.eye(2)
        
        metric = metric_from_jacobian(J, 2)
        
        # Metric should be identity matrix
        expected = sp.eye(2)
        assert metric.equals(expected)
        
    def test_2D_polar_metric(self):
        """Test metric for 2D polar coordinates."""
        r, theta = sp.symbols('r theta', real=True)
        
        # Jacobian for polar coordinates
        # x = r*cos(theta), y = r*sin(theta)
        J = sp.Matrix([
            [sp.cos(theta), -r*sp.sin(theta)],
            [sp.sin(theta), r*sp.cos(theta)]
        ])
        
        metric = metric_from_jacobian(J, 2)
        
        # Expected metric for polar coordinates
        # g_rr = 1, g_rθ = g_θr = 0, g_θθ = r^2
        assert sp.simplify(metric[0, 0] - 1) == 0
        assert sp.simplify(metric[0, 1]) == 0
        assert sp.simplify(metric[1, 0]) == 0
        assert sp.simplify(metric[1, 1] - r**2) == 0


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_cartesian_to_polar_full_pipeline(self):
        """Test full pipeline from coordinate transformation to numerical coefficients."""
        # Define polar to Cartesian transformation
        def to_cartesian(r, theta):
            return r * sp.cos(theta), r * sp.sin(theta)
            
        # Get kinetic energy
        derivatives, coords = get_kinetic_energy(to_cartesian, m=1)
        
        # Vectorize coefficients
        vectorized = vectorize_coefficients(derivatives, coords)
        
        # Test with specific values
        r_vals = np.array([[1, 2, 3]])
        theta_vals = np.array([[0, np.pi/2, np.pi]])
        
        # Evaluate all coefficients
        for order, func in vectorized.items():
            result = func(*np.meshgrid(r_vals, theta_vals, indexing='ij'))
            assert result.shape == (3, 3)
            assert np.all(np.isfinite(result))
            
    def test_1D_harmonic_oscillator_setup(self):
        """Test setting up a 1D harmonic oscillator."""
        # Cartesian coordinates (no transformation)
        def to_cartesian(x):
            return x,
            
        # Get kinetic energy
        derivatives, coords = get_kinetic_energy(to_cartesian, m=1)
        
        # Add potential energy term
        x = coords[0]
        derivatives[(0,)] = derivatives.get((0,), 0) + x**2 / 2
        
        # Vectorize
        vectorized = vectorize_coefficients(derivatives, coords)
        
        # Test evaluation
        x_vals = np.linspace(-2, 2, 10)
        
        # Kinetic term
        kinetic = vectorized[(2,)](x_vals)
        assert np.all(kinetic == -0.5)
        
        # Potential term
        potential = vectorized[(0,)](x_vals)
        np.testing.assert_array_almost_equal(potential, x_vals**2 / 2)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])

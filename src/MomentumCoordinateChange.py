import inspect
import numpy as np
import sympy as sp

def extract_derivatives(expr, vars, function, derivatives=None, coeff=1):
    """
    Recursively extract derivatives from a symbolic expression.
    
    Args:
        expr: Symbolic expression to analyze
        vars: List of variables
        function: Function to extract coefficients from
        derivatives: Dictionary to store derivative coefficients
        coeff: Current coefficient multiplier
        
    Returns:
        Dictionary mapping derivative orders to their coefficients
    """

    if derivatives is None:
        derivatives = {}
    
    # If the expression is a lone derivative, 
    # then `coeff` contains the coefficient to that derivative term 
    if isinstance(expr, sp.Derivative):
        order = [0] * len(vars)
        for v in expr.variables:
            order[vars.index(v)] += 1
        order = tuple(order)
        derivatives[order] = derivatives.get(order, 0) + coeff * expr.expr.coeff(function) 

    # If the expression is an addition of derivative terms, extract the coefficient of each term 
    elif isinstance(expr, sp.Add):
        for term in expr.args:
            extract_derivatives(term, vars, function, derivatives, coeff)

    # If the expression is a multiplication statement, 
    # then it should look like a coefficient multiplying a derivative or a constant term. 
    elif isinstance(expr, sp.Mul):
        new_coeff = coeff
        new_expr = 1
        has_derivative = False
        for factor in expr.args:
            if isinstance(factor, sp.Derivative):
                new_expr *= factor
                has_derivative = True
            else:
                new_coeff *= factor
        if has_derivative:
            extract_derivatives(new_expr, vars, function, derivatives, new_coeff)
        else:
            order = tuple([0] * len(vars))
            derivatives[order] = derivatives.get(order, 0) + new_coeff/function

    return derivatives

def get_kinetic_energy(to_cartesian, m=1):
    """
    Calculate the kinetic energy operator in transformed coordinates.
    
    Args:
        to_cartesian: Function mapping transformed coordinates to Cartesian coordinates
        m: Mass (default=1)
        
    Returns:
        Tuple of (derivatives dictionary, transformed coordinate symbols)
    """
    # Dynamically find the number of dimensions in the transformation
    sig = inspect.signature(to_cartesian)
    n_dims = len(sig.parameters)

    # Create the transformed coordinate vector
    X_tilde = sp.Matrix(sp.symbols(f'x(0:{n_dims})_tilde', real=True))

    # Apply the transformation to find the cartesian coordinates
    X = sp.Matrix(to_cartesian(*X_tilde))
    
    # Jacobian matrix of the transformation
    J = X.jacobian(X_tilde)
    
    # Determinant of the Jacobian matrix
    det_J = J.det()
    root_det_J = sp.sqrt(det_J)
    
    # Inverse of the Jacobian matrix for the \frac{\partial \tilde{x}_j}{\partial x_i} coefficient
    J_inv = J.inv()

    # Define momentum-generating funciton
    def momentum(i):
        return lambda psi: -sp.I * root_det_J * sp.Add(*[J_inv[j, i] * 
                                sp.diff(psi/root_det_J, X_tilde[j]) for j in range(n_dims)])
    
    # Define momentum operators for each dimension
    p = []
    for i in range(n_dims):
        p.append(momentum(i))
    
    # Construct the Hamiltonian operator (for simplicity, assume a free particle H = p^2 / 2m)
    psi = sp.Function('psi')(*X_tilde)
    p_2 = [pi(pi(psi)) for pi in p]
    
    # Hamiltonian in Cartesian coordinates
    K = sp.Add(*[p_2[i] for i in range(n_dims)]) / (2 * m)
    K = K.simplify()  # Good practice to simplify complex expressions

    # Extract all derivatives
    derivatives = {}
    extract_derivatives(K.expand(), list(X_tilde), psi, derivatives)
    return derivatives, X_tilde


def metric_from_jacobian(J, n_dims):
    def matrix_construction_function(i, j):
        I = sp.eye(n_dims)
        terms = []
        for a in range(n_dims):
            for b in range(n_dims):
                terms.append(J[a, i] * J[b, j] * I[a, b])
        
        return sp.Add(*terms)
    
    return sp.Matrix(n_dims, n_dims, matrix_construction_function)
    
def get_kinetic_energy_bundle(to_cartesian, m=1):
    """
    Calculate the kinetic energy operator in transformed coordinates.
    
    Args:
        to_cartesian: Function mapping transformed coordinates to Cartesian coordinates
        m: Mass (default=1)
        
    Returns:
        Tuple of (derivatives dictionary, transformed coordinate symbols)
    """
    # Dynamically find the number of dimensions in the transformation
    sig = inspect.signature(to_cartesian)
    n_dims = len(sig.parameters)

    # Create the transformed coordinate vector
    X_tilde = sp.Matrix(sp.symbols(f'x(0:{n_dims})_tilde', real=True))

    # Apply the transformation to find the cartesian coordinates
    X = sp.Matrix(to_cartesian(*X_tilde))
    
    # Jacobian matrix of the transformation
    J = X.jacobian(X_tilde)

    metric = metric_from_jacobian(J, n_dims)
    
    # Determinant of the Jacobian matrix
    det_J = J.det()
    #root_det_J = sp.sqrt(det_J)
    
    # Inverse of the Jacobian matrix for the \frac{\partial \tilde{x}_j}{\partial x_i} coefficient
    J_inv = J.inv()

    # Define momentum-generating funciton
    def momentum(i):
        return lambda psi: -sp.I * (sp.diff(psi, X_tilde[i]) + psi*sp.diff(sp.ln(det_J), X_tilde[i])/2)
    
    # Define momentum operators for each dimension
    p = []
    for i in range(n_dims):
        p.append(momentum(i))
    
    # Construct the Hamiltonian operator (for simplicity, assume a free particle H = p^2 / 2m)
    psi = sp.Function('psi')(*X_tilde)
    p_2 = [metric[i, i] * pi(pi(psi)) for i, pi in enumerate(p)]
    
    # Hamiltonian in Cartesian coordinates
    K = sp.Add(*[p_2[i] for i in range(n_dims)]) / (2 * m)
    K = K.simplify()  # Good practice to simplify complex expressions

    # Extract all derivatives
    derivatives = {}
    extract_derivatives(K.expand(), list(X_tilde), psi, derivatives)
    return derivatives, X_tilde

def vectorize_coefficients(derivatives, transformed_coordinates):
    """
    Convert symbolic coefficients to vectorized numerical functions.
    
    Args:
        derivatives: Dictionary of derivative orders and symbolic coefficients
        transformed_coordinates: List of coordinate symbols
    """
    vectorized_derivatives = {}
    for order, coeff in derivatives.items():
        vectorized_derivatives[order] = np.vectorize(sp.lambdify(transformed_coordinates, coeff.simplify()))
    return vectorized_derivatives

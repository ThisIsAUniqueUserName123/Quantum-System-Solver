import numpy as np
from src.LinearOperatorSystem import LinearOperatorSystem
from src.BoundaryConditions import zero_boundary_condition 

# Define the coordinate grid 
cartesian_divisions = [1000]
cartesian_bounds = [(-5, 5)]  # Domain boundaries

# Build the cartesian system 
cartesian_system = LinearOperatorSystem(
    bounds=cartesian_bounds,
    num_divisions=cartesian_divisions,
    operator_dict={
        (2,): lambda x: -np.ones_like(x)/2,
    },
    boundary_conditions=[zero_boundary_condition(0)]
)

# Find the ground state solution
_, psi_cartesian = cartesian_system.compute_eigenstates(1)
cartesian_system.plot_eigenstates(psi_cartesian, 1, complex_tool=lambda x: np.abs(x)**2)
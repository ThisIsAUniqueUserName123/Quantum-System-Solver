import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs

class LinearOperatorSystem:
    def __init__(self, bounds, num_divisions, operator_dict, boundary_conditions, accuracy_order=2):
        """
        Initialize a quantum system with specified parameters.
        
        Args:
            bounds: List of tuples with min and max bounds for each dimension
            num_divisions: Number of grid points in each dimension
            operator_dict: Dictionary mapping derivative orders to coefficient functions
            boundary_conditions: List of boundary condition functions
            accuracy_order: Order of accuracy for finite difference approximation
        """
        self.bounds = bounds
        self.num_divisions = num_divisions
        self.operator_dict = operator_dict
        self.boundary_conditions = boundary_conditions
        self.accuracy_order = accuracy_order
        self.num_dims = len(bounds)
        self.grid_points = [np.linspace(bounds[i][0], bounds[i][1], num_divisions[i]) for i in range(self.num_dims)]
        self.grid = np.array(np.meshgrid(*self.grid_points, indexing='ij'))
        self.dx = [(bounds[i][1] - bounds[i][0]) / (num_divisions[i] - 1) for i in range(self.num_dims)]
        self.N = np.prod(num_divisions)
        self.operator = self._create_finite_operator()

    def get_kernel_1D(self, order, accuracy_order=1):
        """
        Generate 1D finite difference kernel for specified derivative order and accuracy.
        
        Args:
            order: Order of the derivative
            accuracy_order: Order of accuracy for the finite difference scheme
            
        Returns:
            Tuple of (finite difference weights, center index)
        """
        if accuracy_order < order:
            raise ValueError(f'Cannot construct stencil of order {order} with accuracy = {accuracy_order}')

        N = accuracy_order + 1 - ((order+1)%2)*(accuracy_order%2)
        center = (N-1)//2
        v = np.arange(N) - center
        V = np.array([v**i for i in range(N)])
        b = np.zeros(N)
        b[order] = factorial(order)
        finite_scheme = np.linalg.solve(V, b)
        return finite_scheme, center

    def _get_kernel(self, order):
        """
        Create an N-dimensional kernel by adding stencils.
        
        Args:
            order: List of derivative orders for each dimension
            
        Returns:
            Tuple of (kernel array, center indices)
        """

        # Collect all 1D kernels and their centers
        kernel_dims, centers = zip(*[self.get_kernel_1D(o, self.accuracy_order) for o in order])
        centers = np.array(centers, dtype=int)

        # Build the ND kernel
        kernel = 1
        for i, o in enumerate(order):
            kernel_dim = kernel_dims[i] / self.dx[i]**o 
            kernel = np.multiply.outer(kernel, kernel_dim)
        return kernel, centers
        
    def _finite_difference_from_kernel(self, kernel, centers):
        """
        Create a finite difference matrix from a kernel.
        
        Args:
            kernel: Finite difference kernel weights
            centers: Center indices for the kernel
            
        Returns:
            Sparse matrix for finite difference operator
        """
    
        # Initialize a sparse matrix in lil-format
        finite_difference = lil_matrix((self.N, self.N), dtype=complex)

        # Helper method for converting to flat indexing
        def multi_to_flat_index(index):
            return np.ravel_multi_index(index, self.num_divisions)

        # Construct all indices for the kernel
        kernel_idx = [np.arange(s) for s in kernel.shape]
        kernel_grid = np.meshgrid(*kernel_idx, indexing='ij')
        kernel_outer_grid = [np.add.outer(kernel_grid[i], np.zeros(self.N, dtype=int)) for i in range(self.num_dims)]

        # Construct all indices for the matrix
        diagonal_indices = np.unravel_index(np.arange(self.N), self.num_divisions)
        matrix_first_index = [np.add.outer(np.zeros(kernel_grid[i].shape, dtype=int), diagonal_indices[i]) for i in range(self.num_dims)]
        matrix_second_idx = [np.add.outer(kernel_grid[i] - centers[i], diagonal_indices[i]) for i in range(self.num_dims)]

        # The boundary conditions transform the indices
        # They also define what indices (if any) to discard 
        # Discarding indices corresponds to assuming that entry is exactly zero. 
        for j, bc_func in enumerate(self.boundary_conditions):
            # Call the boundary condition function
            indices_grid, retention_mask = bc_func(matrix_second_idx, self.num_divisions) 

            # Apply the retention mask to each index grid
            if retention_mask is not None:
                kernel_outer_grid = [k[retention_mask] for k in kernel_outer_grid]
                matrix_first_index = [m[retention_mask] for m in matrix_first_index]
                matrix_second_idx = [m[retention_mask] for m in matrix_second_idx]

        # Convert to linear indexing
        matrix_first_index = multi_to_flat_index(matrix_first_index)
        matrix_second_idx = multi_to_flat_index(matrix_second_idx)

        # Sparse arrays only accept 1D indexing, so we flatten each indexing
        matrix_first_index = matrix_first_index.flatten()
        matrix_second_idx = matrix_second_idx.flatten()
        kernel_outer_grid = [sgg.flatten() for sgg in kernel_outer_grid]
        
        # Add the full kernel to the matrix
        finite_difference[matrix_first_index, matrix_second_idx] += kernel[*kernel_outer_grid]

        return finite_difference.tocsc()

    def _get_finite_difference_stencil(self, order):
        """
        Generate a finite difference stencil for specified derivative orders.
        
        Args:
            order: List of derivative orders for each dimension
            additive: Whether to use additive or multiplicative stencil construction
            
        Returns:
            Sparse matrix for finite difference operator
        """
        
        kernel, centers = self._get_kernel(order)
        return self._finite_difference_from_kernel(kernel, centers)

    def _create_finite_operator(self):
        """
        Create the operator matrix from the specified derivatives and coefficients.
        
        Returns:
            Sparse matrix representing the operator
        """
        operator = lil_matrix((self.N, self.N), dtype=complex)

        # Construct the full operator by multiplying each functional coefficient 
        # by its corresponding derivative operator matrix
        for deriv_order, coeff_func in self.operator_dict.items():
            if len(deriv_order) != self.num_dims:
                raise ValueError("Derivative order must match the number of dimensions")
    
            stencil = self._get_finite_difference_stencil(deriv_order)
            coefficients = coeff_func(*self.grid).flatten()
            operator += stencil.multiply(coefficients[:, np.newaxis]) 

        return operator.tocsc()

    def compute_eigenstates(self, k):
        """
        Compute the lowest k eigenstates of the operator.
        
        Args:
            k: Number of eigenstates to compute
            
        Returns:
            Tuple of (eigenvalues, normalized eigenvectors)
        """
        eigenvalues, eigenvectors = eigs(self.operator, k=k, which='SR')

        dnx = np.prod(self.dx)
        normalization_factor = np.sqrt(np.sum(np.abs(eigenvectors)**2, axis=0)*dnx)
        eigenvalues /= normalization_factor
        eigenvectors /= normalization_factor
        
        return eigenvalues, eigenvectors.reshape((*self.num_divisions, k))

    def plot_eigenstates(self, eigenvectors, k, slice_axis=[], index=[], complex_tool = np.real):
        """
        Plot eigenstates for 1D or 2D systems or slices of higher-dimensional systems.
        
        Args:
            eigenvectors: Eigenvectors to plot
            k: Number of eigenstates to plot
            slice_axis: Dimensions to slice
            index: Indices for slicing each dimension
        """
        # Ensure both the slice axis and index are iterable objects
        try:
            iter(slice_axis)
        except TypeError:
            slice_axis = [slice_axis]
    
        try:
            iter(index)
        except TypeError:
            index = [index]

        # Make sure the slice axes and indices correspond
        if len(slice_axis) != len(index):
            raise ValueError("Number of slice axis must correspond to the number of slice indices")

        # Construct a list of all axes and remove those specified by slice_axis and index
        axes_to_keep = [slice(s) for s in eigenvectors.shape]
        indices_to_keep = list(range(len(eigenvectors.shape)-1))
        
        for i, axis in enumerate(slice_axis):
            axes_to_keep[axis] = index[i]
            indices_to_keep.remove(axis)

        # Construct the solution slice
        solution = eigenvectors[tuple(axes_to_keep)]
        solution_bounds = [self.bounds[i] for i in indices_to_keep]
        solution_divs = [self.num_divisions[i] for i in indices_to_keep]

        # Plot the solutions
        num_dims = len(eigenvectors.shape) - 1 - len(slice_axis)
        if num_dims == 1:
            for i in range(k):
                plt.plot(np.linspace(*solution_bounds[0], solution_divs[0]), complex_tool(solution[..., i]))
            plt.xlabel('x')
            plt.ylabel('Eigenstate')
            plt.title(f'First {k} Eigenstates')
            plt.show()
        elif num_dims == 2:
            for i in range(k):
                plt.imshow(complex_tool(solution[..., i]).T, extent=[*solution_bounds[0], *solution_bounds[1]], origin="lower")
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(f'Eigenstate {i+1}')
                plt.colorbar()
                plt.show()
        else:
            raise ValueError("Only 1D and 2D systems are supported for plotting")

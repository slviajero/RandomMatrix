# Random Matrix Simulation Class 
#
# The original code was intended for the simulation of the fixed
# scattterer problem. This is now separated into a class for random
# matrices specifically for the analysis of general distance matrices
# and uncorrelated random matrices that resemble distance matrices.
#
# The class currently supports numpy and cupy as solvers.
# jax is planned and only stubs are implemented.
#
# required modules:
# numpy, scipy, matplotlib, time

import numpy as np
from scipy.linalg import eigh
from scipy.stats import gaussian_kde    
import matplotlib.pyplot as plt
import time
import sys

# can we accellerate the code with cupy?
try:
    import cupy as cp
    cupy_available = True
except ImportError:
    cupy_available = False

# or even better with jax?
try:
    import jax
    import jax.numpy as jnp
    jax_available = True
except ImportError:
    jax_available = False


# The matrix class is a base class for creating different types of matrices.
# a matrix ensemble as to implement/overload the setup method. This method
# must populate self.matrix with the matrix data. The sample method 
# calls setup once and then calculates a number of observables from the matrix.

class Matrix:
    def __init__(self, N, solver='auto'):
        self.solver = solver
        self.N = N
        self.eigenvalues = None
        self.eigenvectors = None
        self.element_sum = None
        self.element_sumsquare = None
        self.histogram_values = None
        self.eigenvalue_median = None
        self.row_max = None
        self.row_min = None
        self.eigenvectorsigns = None
        self.zerodiag = True
        self.projections = None
        self.confcount = 0
        self.nearest_neighbor = None
        self.nearest_neighbor_total = 0.0
        self.maximum_distance = 0.0

        # determine the solver to use
        if solver == 'auto':
            if cupy_available:
                self.solver = 'cupy'
            elif jax_available:
                self.solver = 'jax'
            else:
                self.solver = 'numpy'
        elif solver not in ['numpy', 'cupy', 'jax']:
            raise ValueError("Invalid solver. Choose 'numpy', 'cupy', or 'jax'.")
        if self.solver == 'cupy' and not cupy_available:
            print("Warning: cupy is not available. Falling back to numpy.")
            self.solver = 'numpy'
        if self.solver == 'jax' and not jax_available:
            print("Warning: jax is not available. Falling back to numpy.")
            self.solver = 'numpy'
 
        # initialize the matrix and projections with zeros to create the objects
        if self.solver == 'numpy':
            self.matrix = np.zeros((self.N, self.N))
            self.projections = np.zeros(self.N)
            self.projection_vector = np.ones(self.N)
        elif self.solver == 'cupy':
            self.matrix = cp.zeros((self.N, self.N))
            self.projections = cp.zeros(self.N)
            self.projection_vector = cp.ones(self.N)
        elif self.solver == 'jax':
            print("Warning: jax is not fully supported. Using numpy.")
            self.solver = 'numpy'
        else:
            raise ValueError("Invalid solver. Choose 'numpy' or 'cupy'.")

    def setup(self, param1, param2):
        pass

    def sample(self, param1=1.0, param2=0.0):
        self.setup(param1, param2)
        self.eigenvectorsigns = []
        self.projections = []
        if self.solver == 'cupy':
            self.element_sum = cp.asnumpy(cp.sum(self.matrix))
            self.element_sumsquare = cp.asnumpy(cp.sum(cp.square(self.matrix)))
            rowssum = cp.sum(self.matrix, axis=1)
            nearest = cp.max(self.matrix, axis=1)
            self.nearest_neighbor_total = cp.asnumpy(1 / cp.max(self.matrix))
            self.row_max = cp.asnumpy(cp.max(rowssum))
            self.row_min = cp.asnumpy(cp.min(rowssum))
            self.nearest_neighbor = cp.asnumpy(cp.mean(1 / nearest))
            self.eigenvalues, self.eigenvectors = cp.linalg.eigh(self.matrix)
            projections = cp.asnumpy(cp.dot(self.eigenvectors.T, self.projection_vector))
            signs = cp.asnumpy(cp.mean(cp.sign(self.eigenvectors), axis=0)) * self.N
            self.projections.extend(projections.tolist())
            self.eigenvectorsigns.extend(signs.tolist())
            self.eigenvectors = cp.asnumpy(self.eigenvectors)
            self.eigenvalues = cp.asnumpy(self.eigenvalues)
        else:
            self.element_sum = np.sum(self.matrix)
            self.element_sumsquare = np.sum(np.square(self.matrix))
            rowsum = np.sum(self.matrix, axis=1)
            nearest = np.max(self.matrix, axis=1)
            self.row_max = np.max(rowsum)
            self.row_min = np.min(rowsum)
            self.nearest_neighbor_total = 1 / np.max(self.matrix)
            self.nearest_neighbor = np.mean(1 / nearest)
            self.eigenvalues, self.eigenvectors = eigh(self.matrix, overwrite_a=True)
            projections = np.dot(self.eigenvectors.T, self.projection_vector)
            signs = np.mean(np.sign(self.eigenvectors), axis=0) * self.N
            self.projections.extend(projections.tolist())
            self.eigenvectorsigns.extend(signs.tolist())

        self.confcount += 1

# Uncorrelated matrices with the same distributions as the distance matrices
# Currently only implemented on numpy and only for the normal distribution.

class MatrixUncorrelated(Matrix):
    def __init__(self, N, scaled=False, distribution='normal', hardcore=0.001):
        super().__init__(N)
        self.scaled = scaled
        self.hardcore = hardcore
        self.hardcore_count = 0.0
        self.distribution = distribution

    def setup(self, param1=1.0, param2=0.0):
        self.matrix = np.zeros((self.N, self.N))
        self.zerodiag = True
        self.cutoff = 1 / self.hardcore

        # scale the stadard deviation with n^(1/3)
        if self.scaled:
            param1 = param1 * (self.N/4.0)**(1.0/3.0)

        if self.distribution == 'normal':
            # Generate all pairwise differences using broadcasting for N*(N-1)/2 pairs
            values1 = np.random.normal(0, param1, size=(self.N * (self.N - 1) // 2, 3))
            values2 = np.random.normal(0, param1, size=(self.N * (self.N - 1) // 2, 3))
        else:
            raise ValueError("Invalid distribution. Only 'normal' is supported for this matrix type.")
        values = np.linalg.norm(values1 - values2, axis=1)
        values = 1 / values
        values[np.isinf(values)] = self.cutoff  # set inf to cutoff
        values[np.isnan(values)] = self.cutoff  # set nan to cutoff
        values[values > self.cutoff] = self.cutoff  # set values above cutoff
        self.hardcore_count += np.sum(values > self.cutoff)
        # Fill upper triangle with values using numpy broadcasting
        triu_indices = np.triu_indices(self.N, k=1)
        self.matrix[triu_indices] = values
        self.matrix[(triu_indices[1], triu_indices[0])] = values  # Symmetrize
        np.fill_diagonal(self.matrix, 0)


# a few test ensembles, implemented only on numpy for now.

# Test matrix to simulate the GOE ensemble of Gaussian matrices.
# Symmetric Gaussian matrix.

class MatrixGaussian(Matrix):
    def setup(self, param1=1.0, param2=0.0):
        self.zerodiag = True
        # Fill upper triangle (excluding diagonal) with random values, then symmetrize and zero diagonal
        upper_indices = np.triu_indices(self.N, k=1)
        values = np.random.normal(param2, param1, size=len(upper_indices[0]))
        self.matrix[upper_indices] = values
        self.matrix[(upper_indices[1], upper_indices[0])] = values  # Symmetrize
        np.fill_diagonal(self.matrix, 0)

# Symmetric uniform matrix
class MatrixUniform(Matrix):
    def setup(self, param1=1.0, param2=0.0):
        lower_bound = param2 - param1/2
        upper_bound = param2 + param1/2
        self.zerodiag = True
        # Fill upper triangle (excluding diagonal) with random values, then symmetrize and zero diagonal
        upper_indices = np.triu_indices(self.N, k=1)
        values = np.random.uniform(lower_bound, upper_bound, size=len(upper_indices[0]))
        self.matrix[upper_indices] = values
        self.matrix[(upper_indices[1], upper_indices[0])] = values  # Symmetrize
        np.fill_diagonal(self.matrix, 0)

# to test average and variance of the matrix elements
class MatrixUnity(Matrix):
    def setup(self, param1=1.0, param2=0.0):
        self.matrix = np.ones((self.N, self.N))
        self.zerodiag = True
        np.fill_diagonal(self.matrix, 0)
  
# Samples (inverse) distances for various distributions in a 
# space of dimension dimension. Mostly used and tested right 
# now for 3D. 
#
# Default are inverse distance i.e. correlated matrixed of the 
# shape 1 / | x_i - x_j |^exponent where x_i and x_j are positions
# in an euclidean space. exponent can be set to any value. For this
# reason the class can be used for distance matrices as well. 
#
# Code is using broadcasting for all operations and should be fast
# on GPUs.
#
# Parameters:
# - N: number of particles
# - scaled: if True, the standard length is scaled with n^(1/3), i.e.
#          the density of particles remains constant.
# - distribution: the distribution of the particles, can be 'normal', 'cube',
#                 'uniform', 'student_t', 'sphere', 'circle', 'line', 'plane'.
# - hardcore: the hardcore distance, i.e. the minimum distance between particles
#                to avoid singularities in the distance matrix. 
# - solver: the solver to use, can be 'numpy', 'cupy', or 'jax'. Default is 'auto'.
# - correlated: currently unused
# - exponent: the exponent to use for the distance matrix, default is -1.0.
# - logarithmic: if True, the distance matrix is calculated as -log(distance * logscale).
# - logscale: the scale for the logarithmic distance matrix, default is 1.0.
# - dimension: the spacial dimension of the space, works for all distributions that
#                are not based on polar coordinates.

class MatrixInverseDistance(Matrix):
    def __init__(self, 
                N, 
                scaled=True, 
                distribution='normal', 
                hardcore=0.001, 
                solver='auto', 
                correlated=True,
                exponent=-1.0, 
                logarithmic=False,
                logscale=1.0,
                dimension=3):
        super().__init__(N, solver=solver)
        self.scaled = scaled
        self.distribution = distribution
        self.hardcore = hardcore
        self.hardcore_count = 0.0
        self.manifold_dimension = dimension
        self.space_dimension = dimension
        self.exponent = exponent
        self.logarithmic = logarithmic
        self.logscale = logscale,
        self.correlated = correlated
        self.cutoff = 1 / self.hardcore # only for exponent -1 

        print("MatrixInverseDistance initialized with parameters:"
              "\nN: {}, scaled: {}, distribution: {}, hardcore: {}, "
              "solver: {}, correlated: {}, exponent: {}, logarithmic: {}, "
              "logscale: {}, space_dimension: {}".format(
                  self.N, self.scaled, self.distribution, self.hardcore,
                  self.solver, self.correlated, self.exponent, self.logarithmic,
                  self.logscale, self.space_dimension))

        if self.space_dimension < 2:
            raise ValueError("Distribution not supported for space dimension {}: {}".format(self.space_dimension, self.distribution))

        # hope the best
        #if not self.correlated:
        #    raise ValueError("Uncorrelated matrices not yet supported. Use class MatrixUncorrelated instead.")

    # generate a set of coordinates samples from the specified distribution
    def configuration(self, param1=1.0, param2=0.0, size=None):
        pass

    # normal distribution of coordinates 
    def configuration_normal(self, param1=1.0, param2=0.0, size=None):
        if size is None:
            size = self.N
        if self.solver == 'cupy':
            import cupy as cp
            self.coords = cp.random.normal(param2, param1, size=(size, self.space_dimension))
        else:
            self.coords = np.random.normal(param2, param1, size=(size, self.space_dimension))
        return self.coords
    
    # homogeneous cube distribution of coordinates
    def configuration_cube(self, param1=1.0, param2=0.0, size=None):
        if size is None:
            size = self.N
        lower_bound = param2 - param1
        upper_bound = param2 + param1
        if self.solver == 'cupy':
            import cupy as cp
            self.coords = cp.random.uniform(lower_bound, upper_bound, size=(size, self.space_dimension))
        else:
            self.coords = np.random.uniform(lower_bound, upper_bound, size=(size, self.space_dimension))
        return self.coords
    
    # a general distribution for radial coordinates in 3 dimensions
    # not used anymore, but kept for reference
    def configuration_radial3d(self, r = 1.0, size=None):
        if size is None:
            size = self.N
        if self.solver == 'cupy':
            import cupy as cp
            theta = cp.random.uniform(0, 2 * cp.pi, size)
            phi = cp.arccos(cp.random.uniform(-1, 1, size)) 
            x = r * cp.sin(phi) * cp.cos(theta)
            y = r * cp.sin(phi) * cp.sin(theta)
            z = r * cp.cos(phi)
            self.coords = cp.column_stack((x, y, z))
        else:
            theta = np.random.uniform(0, 2 * np.pi, size)
            phi = np.arccos(np.random.uniform(-1, 1, size)) 
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            self.coords = np.column_stack((x, y, z))
        return self.coords

    # generalization to hyperspheres in arbitrary dimensions
    def configuration_hypersphere(self, r=1.0, size=None):
        if size is None:
            size = self.N
        if self.solver == 'cupy':
            vec = cp.random.randn(size, self.space_dimension)
            norms = cp.linalg.norm(vec, axis=1, keepdims=True)
            vecs_normalized = vec / norms
        else:
            vec = np.random.randn(size, self.space_dimension)
            norms = np.linalg.norm(vec, axis=1, keepdims=True)
            vecs_normalized = vec / norms
            # print("vecs_normalized shape:", vecs_normalized.shape)
        self.coords = vecs_normalized * r.reshape(-1, 1)  # scale by radius r
        # print("self.coords shape:", self.coords.shape)
        return self.coords
    
    # a uniform distribution of coordinates in a sphere
    def configuration_uniform(self, param1=1.0, param2=0.0, size=None):
        if size is None:
            size = self.N
        if self.solver == 'cupy':
            r = param1 * cp.cbrt(cp.random.uniform(0, 1, size))            
        else:
            r = param1 * np.cbrt(np.random.uniform(0, 1, size))
        # 3d code
        #self.configuration_radial3d(r=r, size=size)
        self.configuration_hypersphere(r=r, size=size)
        return self.coords
    
    # a Student's t-distribution of coordinates
    # Note: the degrees of freedom must be greater than 2 for the distribution to be valid
    def configuration_student_t(self, param1=1.0, param2=0.0, size=None):
        if size is None:
            size = self.N
        if param2 < 0:
            raise ValueError("Degrees of freedom must be greater than 2 for Student's t-distribution. Set param2 to a value greater than 0.")
        student_t = param2 + 3  # ensure degrees of freedom is greater than 2
        if self.solver == 'cupy':
            self.coords = cp.random.standard_t(student_t, size=(size, self.space_dimension)) / cp.sqrt(student_t / (student_t - 2)) * param1
        else:
            self.coords = np.random.standard_t(student_t, size=(size, self.space_dimension)) / np.sqrt(student_t / (student_t - 2)) * param1
        return self.coords
    
    # a configuration of coordinates on a circle
    def configuration_circle(self, param1=1.0, param2=0.0, plane=(0,1), size=None):
        if size is None:
            size = self.N
        if self.solver == 'cupy':
            theta = cp.random.uniform(0, 2 * cp.pi, size)
            x = param1 * cp.cos(theta)
            y = param1 * cp.sin(theta)
            self.coords = cp.zeros((size, 3))
            self.coords[:, plane[0]] = x
            self.coords[:, plane[1]] = y
        else:
            theta = np.random.uniform(0, 2 * np.pi, size)
            x = param1 * np.cos(theta)
            y = param1 * np.sin(theta)
            self.coords = np.zeros((size, 3))
            self.coords[:, plane[0]] = x
            self.coords[:, plane[1]] = y
        return self.coords
    
    # particles in a line
    def configuration_line(self, param1=1.0, param2=0.0, size=None):
        if size is None:
            size = self.N
        if self.solver == 'cupy':
            x = cp.random.uniform(-param1, param1, size)
            self.coords = cp.zeros((size, 3))
            self.coords[:, 0] = x
        else:
            x = np.random.uniform(-param1, param1, size)
            self.coords = np.zeros((size, 3))
            self.coords[:, 0] = x
        return self.coords

    # particles in a plane
    def configuration_plane(self, param1=1.0, param2=0.0, size=None):
        if size is None:
            size = self.N
        if self.solver == 'cupy':
            x = cp.random.uniform(-param1, param1, size)
            y = cp.random.uniform(-param1, param1, size)
            self.coords = cp.zeros((size, 3))
            self.coords[:, 0] = x
            self.coords[:, 1] = y
        else:
            x = np.random.uniform(-param1, param1, size)
            y = np.random.uniform(-param1, param1, size)
            self.coords = np.zeros((size, 3))
            self.coords[:, 0] = x
            self.coords[:, 1] = y
        return self.coords
    
    # create the matrix from the coordinates
    def setup(self, param1=1.0, param2=0.0):
        if (self.solver == 'cupy'):
            import cupy as cp
            self.matrix = cp.zeros((self.N, self.N))
        else:
            self.matrix = np.zeros((self.N, self.N))
        self.zerodiag = True

        # scale the stadard deviation with n^(1/3)
        if self.scaled:
            param1 = param1 * (self.N/4.0)**(1.0/self.space_dimension)

        # do we sample correlated matrices? In this case 
        # we need N coordinate vectors in a dimension space.
        # strategy for uncorrelated matrices is to sample
        # N*(N-1)/2 pairs of coordinates and calculate the distances
        # between them. 
        if self.correlated:
            coord_size = self.N
        else:
            coord_size = self.N * (self.N - 1)


        small_distances = []  # to store small distances for checking the matrix validity
        while True:
            # which distribution to use
            if self.distribution == 'normal':
                self.configuration_normal(param1, param2, size=coord_size)
            elif self.distribution == 'cube':
                self.configuration_cube(param1, param2, size=coord_size)
            elif self.distribution == 'uniform':
                self.configuration_uniform(param1, param2, size=coord_size)
            elif self.distribution == 'student_t':
                self.configuration_student_t(param1, param2, size=coord_size)
            elif self.distribution == 'sphere':
                self.manifold_dimension = 2
                r = param1
                self.configuration_hypersphere(r=r, size=coord_size)
            elif self.distribution == 'circle':
                self.manifold_dimension = 1
                r = param1
                self.configuration_circle(r, size=coord_size)
            elif self.distribution == 'line':
                self.manifold_dimension = 1
                self.configuration_line(param1, param2, size=coord_size)
            elif self.distribution == 'plane':
                self.manifold_dimension = 2
                self.configuration_plane(param1, param2, size=coord_size)
            else:
                raise ValueError("Invalid distribution.")
        
            if self.correlated:
                # compute the distance matrix
                if self.solver == 'cupy':
                    self.coords = cp.asarray(self.coords)
                    diff = self.coords[:, cp.newaxis, :] - self.coords[cp.newaxis, :, :]
                    distances = cp.linalg.norm(diff, axis=2)
                    small_distances = cp.asarray(distances[distances > 0])
                    small_distances = small_distances[small_distances < self.hardcore]
                    if len(small_distances) == 0:
                        with cp.errstate(over='ignore', divide='ignore', invalid='ignore'):
                            if self.logarithmic:
                                self.matrix = - cp.log(distances*self.logscale)
                            else:
                                # apply the exponent to the distances
                                if self.exponent == 0:
                                    self.matrix = cp.ones_like(distances)
                                elif self.exponent == 1:
                                    self.matrix = distances
                                elif self.exponent == -1:
                                    self.matrix = 1 / distances
                                else:
                                    self.matrix = cp.power(distances, self.exponent)
                        cp.fill_diagonal(self.matrix, 0)
                        break
                    else:
                        self.hardcore_count += len(small_distances)
                        continue
                else:
                    self.coords = np.asarray(self.coords)
                    diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
                    distances = np.linalg.norm(diff, axis=2)
                    # Find the smallest nonzero distance
                    small_distances = np.asarray(distances[distances > 0])
                    small_distances = small_distances[small_distances < self.hardcore]
                    if len(small_distances) == 0:
                        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                            if self.logarithmic:
                                self.matrix = - np.log(distances*self.logscale)
                            else:
                                # apply the exponent to the distances
                                if self.exponent == 0:
                                    self.matrix = np.ones_like(distances)
                                elif self.exponent == 1:
                                    self.matrix = distances
                                elif self.exponent == -1:
                                    self.matrix = 1 / distances
                                else:
                                    self.matrix = np.power(distances, self.exponent)
                        np.fill_diagonal(self.matrix, 0)
                        break
                    else:
                        self.hardcore_count += len(small_distances)
                        continue
            else:
                # uncorrelated matrices
                # pairwise distances and then fill the matrix, here no 
                # resample on hardcore distances
                if self.solver == 'cupy':
                    values1 = self.coords[0:coord_size // 2, :]
                    values2 = self.coords[coord_size // 2:, :]
                    values = cp.linalg.norm(values1 - values2, axis=1)
                    values = cp.power(values, self.exponent)
                    values[cp.isinf(values)] = self.cutoff  # set inf to cutoff
                    values[cp.isnan(values)] = self.cutoff  # set nan to cutoff
                    values[values > self.cutoff] = self.cutoff  # set values above cutoff
                    self.hardcore_count += cp.sum(values > self.cutoff)
                    # Fill upper triangle with values using numpy broadcasting
                    triu_indices = cp.triu_indices(self.N, k=1)
                    self.matrix[triu_indices] = values
                    self.matrix[(triu_indices[1], triu_indices[0])] = values  # Symmetrize
                    cp.fill_diagonal(self.matrix, 0)
                else:
                    values1 = self.coords[0:coord_size // 2, :]
                    values2 = self.coords[coord_size // 2:, :]
                    values = np.linalg.norm(values1 - values2, axis=1)
                    values = np.power(values, self.exponent)
                    values[np.isinf(values)] = self.cutoff  # set inf to cutoff
                    values[np.isnan(values)] = self.cutoff  # set nan to cutoff
                    values[values > self.cutoff] = self.cutoff  # set values above cutoff
                    self.hardcore_count += np.sum(values > self.cutoff)
                    # Fill upper triangle with values using numpy broadcasting
                    triu_indices = np.triu_indices(self.N, k=1)
                    self.matrix[triu_indices] = values
                    self.matrix[(triu_indices[1], triu_indices[0])] = values  # Symmetrize
                    np.fill_diagonal(self.matrix, 0)
                # we end the loop without resampling
                break

# The Simulation class samples eigenvalues and computes
# element averages and variances from the provided matrix class.
# It takes a matrix instance as input and performs sampling.
# The sample method generates a specified number of samples and computes
# the average and variance of the elements in the matrix.
# The eigenvalues are also stored for further analysis.

class Simulation:
    def __init__(self, Matrix):
        self.Matrix = Matrix
        self.samples = 0
        self.eigenvalues = None
        self.element_sum_list = None
        self.element_sumsquare_list = None
        self.projections = None
        self.element_average = 0.0
        self.element_variance = 0.0
        self.eigenvalue_median = 0.0
        self.unit_vector = np.ones(self.Matrix.N)
        self.storezeroconf = False
        # for the analysis of zero eigenvalues
        self.zeroconf = None
        self.zeroconfthreshold = 0.01
        self.zeroconfeigenvalues = None
        self.zeroconfeigenvectors = None
        self.zeroconfsummary = None
        # the statistics of the rows, max and min at first, more to come
        self.row_max = None
        self.row_min = None
        # signs and nearest neighbour analysis
        self.eigenvectorsigns = None
        self.nearest_neighbor = None

    def random_seed(self, seed):
        # set the random seed for reproducibility
        if self.Matrix.solver == 'cupy':
            cp.random.seed(seed)
        else:
            np.random.seed(seed)

    def sample(self, samples=1000, param1=1.0, param2=0.0):
        self.eigenvalues = []
        self.projections = []
        self.element_sum_list = []
        self.element_sumsquare_list = []
        self.zeroconf = []
        self.zeroconfeigenvalues = []
        self.zeroconfeigenvectors = []
        self.zeroconfsummary = []
        self.row_max = []
        self.row_min = []   
        self.eigenvectorsigns = []
        self.nearest_neighbor = []
        self.nearest_neighbor_total = []
        self.zeroconfprojections = []
        self.samples = samples
        self.param1 = param1
        self.param2 = param2    
 
        for _ in range(samples):
            self.Matrix.sample(param1, param2)
            self.element_sum_list.append(self.Matrix.element_sum)
            self.element_sumsquare_list.append(self.Matrix.element_sumsquare)
            self.eigenvalues.extend(self.Matrix.eigenvalues)
            self.row_max.append(self.Matrix.row_max)
            self.row_min.append(self.Matrix.row_min)
            self.eigenvectorsigns.extend(self.Matrix.eigenvectorsigns)
            # here we use that the eigenvalues are sorted
            # in ascending order, so the biggest eigenvalue is the last one
            # and the rest are the first N-2 ones
            self.projections.extend(self.Matrix.projections)
            # store the zero configuration
            if self.storezeroconf and (np.min(np.abs(self.Matrix.eigenvalues)) < self.zeroconfthreshold):    
                self.zeroconfeigenvalues.append(np.array(self.Matrix.eigenvalues))
                self.zeroconfeigenvectors.append(np.array(self.Matrix.eigenvectors))
                if hasattr(self.Matrix, 'coords'):
                    self.zeroconf.append(np.array(self.Matrix.coords))
            #the average nearest neighbor distance of all rows i.e. particles
            self.nearest_neighbor.append(self.Matrix.nearest_neighbor)
            # the total nearest neighbor distance, i.e. the smallest distance in the matrix
            self.nearest_neighbor_total.append(self.Matrix.nearest_neighbor_total)
 
        #the element pproperties: average and variance with N-1 corrections
        if self.Matrix.zerodiag:
            self.element_average = np.sum(self.element_sum_list) / (self.Matrix.N * (self.Matrix.N - 1)) / samples
            self.element_variance = np.sum(self.element_sumsquare_list) / samples / (self.Matrix.N * (self.Matrix.N - 1)) - self.element_average**2
        else:   
            self.element_average = np.sum(self.element_sum_list) / (self.Matrix.N * self.Matrix.N) /samples
            self.element_variance = np.sum(self.element_sumsquare_list) / samples / (self.Matrix.N * self.Matrix.N) - self.element_average**2

        # element properties
        self.inverse_element_average = 1 / self.element_average
        # various qauntities of the row sums
        self.row_sum_maximum = np.max(self.row_max)
        self.row_sum_maximum_std = np.std(self.row_max) 
        self.row_sum_minimum = np.min(self.row_min)
        self.row_sum_minimum_std = np.std(self.row_min)
        self.row_sum_average = self.element_average * ( self.Matrix.N - 1 )
        self.row_sum_average_std = np.sqrt((self.Matrix.N - 1)* self.element_variance)
        # nearest neighbor code is only for inverse and has to be reworked
        self.nearest_neighbor_average = np.mean(self.nearest_neighbor)
        self.nearest_neighbor_std = np.std(self.nearest_neighbor)
        self.nearest_neighbor_inverse_average = np.mean(1 / np.array(self.nearest_neighbor))
        self.nearest_neighbor_inverse_std = np.std(1 / np.array(self.nearest_neighbor))
        self.nearest_neighbor_total_average = np.mean(self.nearest_neighbor_total)
        self.nearest_neighbor_total_std = np.std(self.nearest_neighbor_total)
        # the zeroconf is a list of configurations with eigenvalues close to zero
        self.analyze_zeroconf(threshold=self.zeroconfthreshold)
        # eigenvalue and eigenvector properties
        # generate a matrix with first index being the configuration
        # and the second index being the eigenvalue, no sort needed 
        # as they are already sorted in the Matrix class
        self.eigenvalues = np.array(self.eigenvalues).reshape(-1, self.Matrix.N)
        self.projections = np.array(self.projections).reshape(-1, self.Matrix.N)
        # Count the number of positive eigenvalues for each configuration
        self.positive = np.sum(self.eigenvalues > 0, axis=1)
        self.negative = np.sum(self.eigenvalues < 0, axis=1)
        self.zero = np.sum(self.eigenvalues == 0, axis=1) # should never happen
        # averages of the eigenvalues 
        self.eigenvalues_averages = np.mean(self.eigenvalues, axis=0)
        self.eigenvalues_median = np.median(self.eigenvalues, axis=0) 
        self.eigenvalues_standard_deviation = np.std(self.eigenvalues, axis=0)

    def print_results(self, file=sys.stdout):

        print("--------------------- Simulation --------------------------")
        print("Sampled matrices:", self.samples, "from", self.Matrix.__class__.__name__, file=file)
        print("  Dimension:", self.Matrix.N, "Distribution:", self.Matrix.distribution, "Correlated:", self.Matrix.correlated, file=file)
        print("  Hard core distance:", self.Matrix.hardcore, "Hard core count:", self.Matrix.hardcore_count, file=file)
        print("  Parameter1:", self.param1, "Parameter2:", self.param2, file=file)
        print("  Exponent:", self.Matrix.exponent, "Logarithmic:", self.Matrix.logarithmic, "Logscale:", self.Matrix.logscale, file=file)
        print("----------------------- Results ----------------------------")
        print("Matrix element average:", 
            self.element_average, "std:", np.sqrt(self.element_variance), file=file)
        print("Inverse of element average:", 1 / self.element_average, file=file)
        print("Average maximum row sum:", 
            self.row_sum_maximum, "std:", self.row_sum_maximum_std, file=file)
        print("Average row sum:", 
              self.row_sum_average, "std:", self.row_sum_average_std, file=file)
        print("Average minimum row sum:", 
            self.row_sum_minimum, "std:", self.row_sum_minimum_std, file=file)
        print("Average nearest neighbor distance:", 
            self.nearest_neighbor_average, "std:", self.nearest_neighbor_std, file=file)
        print("Inverse of average nearest neighbor distance:", 1 / self.nearest_neighbor_average, file=file)
        print("Average of inverse nearest neighbor distance:", 
            self.nearest_neighbor_inverse_average, "std:", self.nearest_neighbor_inverse_std, file=file)
        print("Average smallest distance in matrix:", 
            self.nearest_neighbor_total_average, "std:", self.nearest_neighbor_total_std, file=file)
        print("----------------------- Spectrum ------------------------", file=file)
        if self.storezeroconf:
            print("Average zero conf projection:",
                np.mean(self.square_zeroconfprojections) if self.square_zeroconfprojections else 0,
                "std:", np.std(self.square_zeroconfprojections) if self.square_zeroconfprojections else 0, file=file)

    def zerodensity(self, zrange=None, extrapolation=False):

        if self.eigenvalues is None:
            raise ValueError("Eigenvalues have not been sampled yet. Call the sample method first.")
        
        if zrange is None:
            zrange = self.zeroconfthreshold

        eigenvalues = np.array(self.eigenvalues).flatten()

        if not extrapolation:
            # find the number of eigenvalues in the range [-range, range] around zero
            count = np.sum(np.abs(eigenvalues) <= zrange)
            # calculate the density
            density = count / len(eigenvalues) / (2 * zrange)
            return density, count
        else:
            # for n=2 and n=3 the kde estimate is incorrect 
            # as the distribution is either zero or not continuous, for these 
            # cases we only look at the positive side to identify if kde makes sense
            small_positive_eigenvalues = eigenvalues[np.abs(eigenvalues) < 0.01]
            small_positive_eigenvalues = small_positive_eigenvalues[small_positive_eigenvalues > 0] 
            if len(small_positive_eigenvalues) == 0:
                return 0.0, 0
            else:
                kde = gaussian_kde(eigenvalues, bw_method='scott')
            # calculate the density at zero with kde. Inaccurate for N=3.
                density = kde(0.0)
                zero_density = density[0]

            return zero_density, np.sum(np.abs(eigenvalues) <= zrange)
        

    def analyze_zeroconf(self, threshold=0.01):

        # analyze the zero configuration
        self.zeroconfsummary = []
        self.square_zeroconfprojections = []

        # analyze the zero configuration
        if self.zeroconf is None:
            return
            #raise ValueError("Zero configuration has not been sampled yet. Call the sample method first.")
        
        if len(self.zeroconfeigenvalues) == 0:
            return
            #raise ValueError("No zero configuration found.")
        
        self.zeroconfsummary = []
        self.square_zeroconfprojections = []
        for i in range(len(self.zeroconfeigenvalues)):
            eigenvalues = np.array(self.zeroconfeigenvalues[i])
            eigenvectors = np.array(self.zeroconfeigenvectors[i])
            for j in range(len(eigenvalues)):
                configuration = []
                if np.abs(eigenvalues[j]) < threshold:
                    # the eigenvalue is close to zero
                    # find the corresponding eigenvector
                    eigenvector = eigenvectors[:, j]
                    # calculate the projection of the eigenvector on the unit vector
                    eigenvalue = eigenvalues[j]
                    # the projection of the eigenvector on the unit vector
                    projection = np.dot(self.unit_vector, eigenvector)
                    # the signs of the eigenvector
                    signs = np.sum(np.sign(eigenvector))
                    configuration.append(eigenvalue)
                    configuration.append(projection)
                    configuration.append(signs)
                    configuration.append(eigenvector)
                    if self.zeroconf is not []:
                        configuration.append(self.zeroconf[i])
                    self.zeroconfsummary.append(configuration)
                    self.square_zeroconfprojections.append(np.square(projection))






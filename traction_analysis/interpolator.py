import numpy as np
import pyvista as pv
import scipy as sp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class FluidDataInterpolator:
    def __init__(self, fluid_grid, particle_mesh, exclusion_threshold):
        """
        A functor for interpolating fluid data at any point in space, excluding points inside a particle mesh.

        Args:
            fluid_grid (pv.RectilinearGrid): The fluid data in a rectilinear grid.
            particle_mesh (pv.PolyData): The particle surface mesh.
            exclusion_threshold (float): Distance threshold for excluding points inside the particle.
        """
        self.particle_mesh = particle_mesh
        self.exclusion_threshold = exclusion_threshold

        # Convert Rectilinear Grid to PolyData for distance computation
        self.fluid_poly = fluid_grid.cast_to_unstructured_grid()

        # Precompute distances for the fluid grid points
        self.distances = self.fluid_poly.compute_implicit_distance(particle_mesh, inplace=False)
        valid_indices = np.where(self.distances.point_data['implicit_distance'] > exclusion_threshold)[0]
        self.valid_fluid_mesh = self.fluid_poly.extract_points(valid_indices)

        # Store valid data points and their corresponding fluid data for interpolation
        self.valid_points = self.valid_fluid_mesh.points
        self.pressure = fluid_grid.point_data.get('pressure', None)
        self.velocity = fluid_grid.point_data.get('velocity', None)
        self.viscous_stress = fluid_grid.point_data.get('viscousStressTensor', None)

        # Create Gaussian Process Regressors for each field
        self.gp_regressors = {}
        if self.pressure is not None:
            self.gp_regressors['pressure'] = self.create_gp_regressor(self.pressure[valid_indices])
        if self.velocity is not None:
            self.gp_regressors['velocity'] = self.create_gp_regressor(self.velocity[valid_indices])
        if self.viscous_stress is not None:
            self.gp_regressors['viscousStressTensor'] = self.create_gp_regressor(self.viscous_stress[valid_indices])

    def create_gp_regressor(self, data):
        kernel = RBF(length_scale=3.0) + WhiteKernel(noise_level=0.1)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
        gpr.fit(self.valid_points, data)
        return gpr

    def __call__(self, point, field_name):
        """
        Interpolate the specified fluid data field at a given point.

        Args:
            point (array-like): The 3D point at which to interpolate the fluid data.
            field_name (str): The field to interpolate ("pressure", "velocity", "viscousStressTensor").

        Returns:
            array-like or None: The interpolated value at the point, or None if the point is inside the particle mesh.
        """
        # Compute the distance from the point to the particle mesh
        point_distance = self.particle_mesh.compute_implicit_distance(pv.PolyData([point]), inplace=False)
        if point_distance['implicit_distance'][0] <= self.exclusion_threshold:
            return None  # Point is inside the exclusion zone

        # Interpolate using the appropriate GP regressor
        if field_name in self.gp_regressors:
            return self.gp_regressors[field_name].predict([point])[0]
        else:
            raise ValueError(f"No data available for field '{field_name}'.")

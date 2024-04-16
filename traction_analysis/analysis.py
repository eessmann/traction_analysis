import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyvista as pv
import tqdm as tm
from typing import Callable
import interpolator


class TractionAnalysis:
    def __init__(self, fluid_path, mesh_path):
        """
        Initialize the TractionAnalysis object.

        Parameters:
        - fluid_path (str): Path to the VTK file containing fluid data.
        - mesh_path (str): Path to the VTK file containing mesh data.
        """
        try:
            # Load VTK fluid data
            fluid = pv.read(fluid_path)
            # Get Dimensions of domain
            x_dim, y_dim, z_dim = fluid.dimensions
            # Create shifted periodic images of fluid domain
            fluid_neg_image = fluid.translate((-x_dim, 0, 0))
            fluid_pos_image = fluid.translate((x_dim, 0, 0))
            merged = fluid.merge([fluid_neg_image, fluid_pos_image])

            (x_min, x_max, y_min, y_max, z_min, z_max) = merged.bounds
            x_lin = np.arange(x_min, x_max + 1, 1)
            y_lin = np.arange(y_min, y_max + 1, 1)
            z_lin = np.arange(z_min, z_max + 1, 1)

            # Creating a structured grid
            grid = pv.RectilinearGrid(x_lin, y_lin, z_lin)

            # Save extended fluid domain
            self.fluid = grid.interpolate(merged)
            self.mesh = pv.read(mesh_path)
            self.resampled_mesh = None
            self.gauss_points = np.array([
                [1 / 6, 1 / 6],  # (barycentric coordinates)
                [2 / 3, 1 / 6],
                [1 / 6, 2 / 3]
            ])
            self.weights = np.array([1 / 6, 1 / 6, 1 / 6])
        except Exception as e:
            raise IOError(f"Error reading VTK files: {e}")

    def gaussian_quadrature(self, interpolator: Callable, initial_value: float | np.ndarray):
        # Loop through each face of the mesh
        try:
            integral_value = type(initial_value)()
            for cell in self.mesh.cell:
                face = cell.points

                if len(face) == 3:
                    v0, v1, v2 = face

                    # Area calculation using cross product
                    area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2

                    # Calculate actual positions of the Gauss points on the current triangle
                    transformed_points = v0 + (v1 - v0) * self.gauss_points[:, 0][:, np.newaxis] + (
                                v2 - v0) * self.gauss_points[:, 1][:, np.newaxis]

                    # Calculate integral contribution from this triangle
                    for point, weight in zip(transformed_points, self.weights):
                        value = interpolator(point)
                        integral_value += weight * value * area
        except Exception as e:
            raise ValueError(f"Error during integrator: {e}")
        return integral_value

    def fluid_interpolate(self):
        """
        Project Fluid data onto Particle.

        Returns:
        - TractionAnalysis: self for chaining.
        """
        try:
            cell_fluid = self.fluid
            self.fluid_interpolation()
            self.resampled_mesh = self.mesh.interpolate(cell_fluid)
        except Exception as e:
            raise ValueError(f"Error during fluid interpolation: {e}")
        return self

    def calculate_weights(self):
        """
        Calculate vertex weights for the mesh.

        Returns:
        - TractionAnalysis: self for chaining.
        """
        try:
            if self.resampled_mesh is None:
                raise ValueError("Mesh data has not been resampled with fluid data.")

            weights = np.zeros(self.resampled_mesh.n_points)
            for cell in self.resampled_mesh.cell:
                vec_a = cell.points[1] - cell.points[0]
                vec_b = cell.points[2] - cell.points[0]
                area = 0.5 * np.linalg.norm(np.cross(vec_a, vec_b))
                for vert in cell.point_ids:
                    weights[vert] += area / 3.0
            self.resampled_mesh.point_data["weights"] = weights
        except Exception as e:
            raise ValueError(f"Error calculating weights: {e}")
        return self

    def calculate_traction(self):
        """
        Calculate traction force for the mesh.

        Returns:
        - TractionAnalysis: self for chaining.
        """
        try:
            if self.resampled_mesh is None:
                raise ValueError("Mesh data has not been resampled with fluid data.")

            viscous_stress = np.array(self.resampled_mesh.point_data["viscousStressTensor"]).reshape(-1, 3, 3)
            normals = np.array(self.resampled_mesh.point_normals)
            pressure = np.array(self.resampled_mesh.point_data["pressure"]).reshape(-1, 1, 1)

            press_tensor = np.identity(3) * pressure
            traction_tensor = viscous_stress - press_tensor

            traction = np.einsum('ijk,ik->ij', traction_tensor, normals)
            dev_force = np.einsum('ijk,ik->ij', viscous_stress, normals)
            press_force = np.einsum('ijk,ik->ij', press_tensor, normals)

            self.resampled_mesh.point_data["traction"] = traction
            self.resampled_mesh.point_data["devForce"] = dev_force
            self.resampled_mesh.point_data["pressForce"] = press_force
        except Exception as e:
            raise ValueError(f"Error calculating traction: {e}")
        return self

    def process(self):
        """
        Process the BioFM data.

        Returns:
        - TractionAnalysis: self for chaining.
        """
        self.fluid_interpolate()
        self.calculate_weights()
        self.calculate_traction()
        return self

    def save(self, output_path='processed_mesh.vtp'):
        """
        Save the processed mesh to a file.

        Parameters:
        - output_path (str): Destination path to save the processed mesh. Defaults to 'processed_mesh.vtp'.
        """
        try:
            self.resampled_mesh.save(output_path)
        except Exception as e:
            raise IOError(f"Error saving the processed mesh: {e}")

    def integrate_forces(self):
        """
        Integrate the forces over the mesh.

        Returns:
        - List of total traction, dev force, and press forces
        """
        weights = np.array(self.resampled_mesh.point_data["weights"]).reshape(-1,
                                                                              1)  # Reshape weights to align with forces
        traction = np.array(self.resampled_mesh.point_data["traction"])
        press_force = np.array(self.resampled_mesh.point_data["pressForce"])
        dev_force = np.array(self.resampled_mesh.point_data["devForce"])

        # Use NumPy's sum method to compute the sum along the desired axis
        total_traction = np.sum(weights * traction, axis=0)
        total_press_force = np.sum(weights * press_force, axis=0)
        total_dev_force = np.sum(weights * dev_force, axis=0)

        return total_traction, total_dev_force, total_press_force

    def integrate_torque(self):
        """
        Integrate the torque over the mesh.

        Returns:
        - Total traction torque
        """
        weights = np.array(self.resampled_mesh.point_data["weights"]).reshape(-1,
                                                                              1)  # Reshape weights to align with forces
        traction = np.array(self.resampled_mesh.point_data["traction"])

        # Compute the relative positions of the node to the mesh center
        rel_pos = self.resampled_mesh.points - self.resampled_mesh.center

        # Calculate torques for the mesh
        torques = np.cross(rel_pos, traction)

        # Append torques to vtk file
        self.resampled_mesh.point_data["torque"] = torques

        # Total torque
        total_torque = np.sum(weights * torques, axis=0)
        return total_torque

    def compute_stresslet(self):
        """
        Calculate the stresslet tensor for the traction forces acting on the mesh.
        """
        if self.resampled_mesh is None:
            raise ValueError("Mesh data has not been resampled with fluid data.")

        # Calculate position vectors relative to the geometric center for all points
        rel_pos = self.resampled_mesh.points - self.resampled_mesh.center

        # Extract traction forces for all points
        traction_forces = self.resampled_mesh.point_data['traction']

        # Extract the area (weight) for each point
        areas = self.resampled_mesh.point_data['weights']

        # Calculate the dyadic product of position vectors and traction forces
        # and then weight by area
        # 'ij,ik,i->jk' computes the outer product for each pair of vectors, sums over i (all points)
        # and weights each term by area[i] before summing.
        stresslet = np.einsum('ij,ik,i->jk', rel_pos, traction_forces, areas)

        return stresslet

    def fluid_interpolation(self, exclusion_threshold=0):
        particle_mesh = self.mesh
        fluid_mesh = self.fluid

        # Compute the signed distance field for the particle mesh
        distances = fluid_mesh.compute_implicit_distance(particle_mesh, inplace=True)
        distances.save("/tmp/fluid_test.vtr")
        print("Saving test data to /tmp/fluid_test.vtr")

        valid_indices = np.where(distances.point_data['implicit_distance'] < exclusion_threshold)
        valid_fluid_mesh = fluid_mesh.extract_points(valid_indices)

        # Extract the valid fluid data (you'll need to adjust this based on your data structure)
        valid_fluid_points = valid_fluid_mesh.points
        valid_data_field = fluid_mesh.interpolate(valid_fluid_mesh, radius=exclusion_threshold).point_data['pressure']


def particle_force_timeseries(timesteps, path):
    traction_forces_list = []
    press_forces_list = []
    dev_forces_list = []
    torques_list = []
    stresslet_list = []

    results_dir = path.parent / "converted/"
    results_dir.mkdir(parents=True, exist_ok=True)

    for timestep in tm.tqdm(timesteps, desc="Timestep", position=1, leave=False):
        particle_path = path / f"Particles_t{timestep}.vtp"
        fluid_path = path.parent / "VTKFluid" / f"Fluid_t{timestep}.vtr"
        converted_path = results_dir / f"Particles_t{timestep}.vtp"

        processor = TractionAnalysis(fluid_path, particle_path)
        processor.process()
        traction, dev_force, press_force = processor.integrate_forces()
        torque = processor.integrate_torque()
        stresslet = processor.compute_stresslet()
        traction_forces_list.append(traction)
        dev_forces_list.append(dev_force)
        press_forces_list.append(press_force)
        torques_list.append(torque)
        stresslet_list.append(stresslet)

        processor.save(converted_path)

    # Converting List to Numpy array
    traction_forces = np.array(traction_forces_list)
    dev_forces = np.array(dev_forces_list)
    press_forces = np.array(press_forces_list)
    torques = np.array(torques_list)
    stresslets = np.array(stresslet_list)

    d = {"timestep": timesteps,
         "traction_forces_x": traction_forces[:, 0],
         "traction_forces_y": traction_forces[:, 1],
         "traction_forces_z": traction_forces[:, 2],
         "press_forces_x": press_forces[:, 0],
         "press_forces_y": press_forces[:, 1],
         "press_forces_z": press_forces[:, 2],
         "dev_forces_x": dev_forces[:, 0],
         "dev_forces_y": dev_forces[:, 1],
         "dev_forces_z": dev_forces[:, 2],
         "torques_x": torques[:, 0],
         "torques_y": torques[:, 1],
         "torques_z": torques[:, 2],
         "stresslet_x_x": stresslets[:, 0, 0],
         "stresslet_x_y": stresslets[:, 0, 1],
         "stresslet_x_z": stresslets[:, 0, 2],
         "stresslet_y_x": stresslets[:, 1, 0],
         "stresslet_y_y": stresslets[:, 1, 1],
         "stresslet_y_z": stresslets[:, 1, 2],
         "stresslet_z_x": stresslets[:, 2, 0],
         "stresslet_z_y": stresslets[:, 2, 1],
         "stresslet_z_z": stresslets[:, 2, 2]}
    data = pd.DataFrame(data=d)
    data.to_csv(results_dir / "force_analysis.csv", index=False)
    feather.write_feather(data, results_dir / "force_analysis.fea")
    return data

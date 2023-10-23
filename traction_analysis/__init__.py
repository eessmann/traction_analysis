import pyvista as pv
import numpy as np
import tqdm as tm
import pandas as pd
import argparse as ap
import pathlib
import os
import re


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
        except Exception as e:
            raise IOError(f"Error reading VTK files: {e}")

    def fluid_interpolate(self):
        """
        Project Fluid data onto Particle.

        Returns:
        - TractionAnalysis: self for chaining.
        """
        try:
            cell_fluid = self.fluid
            self.resampled_mesh = self.mesh.sample(cell_fluid)
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
            viscous_stress = np.array(self.resampled_mesh.point_data["viscousStressTensor"]).reshape(-1, 3, 3)
            normals = np.array(self.resampled_mesh.point_normals)
            pressure = np.array(self.resampled_mesh.point_data["pressure"]).reshape(-1, 1, 1)

            press_tensor = np.identity(3) * pressure
            traction_tensor = viscous_stress + press_tensor

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


def is_empty_dir(path: pathlib.Path) -> bool:
    """Check if the directory at the given path is empty."""
    return path.is_dir() and not any(path.iterdir())


# Recursively convert all fluid VTK file into new format
def convert_sim_dirs(input_path: str):
    sim_particle_pattern = re.compile(r"Particles_t(?P<timestep>\d+).vtp")
    sim_root = pathlib.Path(input_path)

    def parse_sim_filename(pattern, key: str, file: str):
        res = pattern.search(file)
        return int(res.group(key))

    def conversion(path, files, pattern, conversion_func):
        # Parse the time steps of the simulation
        time_steps = sorted(list(set([parse_sim_filename(pattern, 'timestep', file) for file in files])))
        # Convert all fluid vtk files in VTKDirectory
        conversion_func(time_steps, path)

    for root, _, files in tm.tqdm(os.walk(input_path), desc="Walking Simulation Directory tree", position=0):
        path = pathlib.Path(root)
        if (os.path.basename(path) == 'VTKParticles') and not is_empty_dir(path):
            conversion(path, files, sim_particle_pattern, particle_force_timeseries)


def particle_force_timeseries(timesteps, path):
    traction_forces_list = []
    press_forces_list = []
    dev_forces_list = []
    for timestep in tm.tqdm(timesteps, desc="Timestep", position=1, leave=False):
        particle_path = path / f"Particles_t{timestep}.vtp"
        fluid_path = path.parent / "VTKFluid" / f"Fluid_t{timestep}.vtr"
        converted_path = path.parent / "converted" / f"Particles_t{timestep}.vtp"

        processor = TractionAnalysis(fluid_path, particle_path)
        processor.process()
        traction, dev_force, press_force = processor.integrate_forces()
        traction_forces_list.append(traction)
        dev_forces_list.append(dev_force)
        press_forces_list.append(press_force)

        processor.save(converted_path)

    # Converting List to Numpy array
    traction_forces = np.array(traction_forces_list)
    dev_forces = np.array(dev_forces_list)
    press_forces = np.array(press_forces_list)

    d = {"timestep": timesteps,
         "traction_forces_x": traction_forces[:, 0], "traction_forces_y": traction_forces[:, 1],
         "traction_forces_z": traction_forces[:, 2],
         "press_forces_x": press_forces[:, 0], "press_forces_y": press_forces[:, 1],
         "press_forces_z": press_forces[:, 2],
         "dev_forces_x": dev_forces[:, 0], "dev_forces_y": dev_forces[:, 1], "dev_forces_z": dev_forces[:, 2]}
    data = pd.DataFrame(data=d)
    data.to_csv(path.parent / "converted" / "force_analysis.csv", index=False)
    data.to_hdf(path.parent / "converted" / "force_analysis.h5", "traction", mode="w")
    return data


# For testing purposes, using the main function
def main():
    parser = ap.ArgumentParser(prog="traction_analysis", description="Particle force analysis script")
    convert_sim_dirs("/home/data/analysis/Simulation/pressure_analysis/pressure_analysis_converted")


# The main function call remains the same
if __name__ == "__main__":
    main()

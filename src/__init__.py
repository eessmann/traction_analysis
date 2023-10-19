import pyvista as pv
import numpy as np


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
            traction = np.empty((0, 3))
            dev_force = np.empty((0, 3))
            press_force = np.empty((0, 3))
            for stress, normal, press in zip(self.resampled_mesh.point_data["viscousStressTensor"],
                                             self.resampled_mesh.point_normals,
                                             self.resampled_mesh.point_data["pressure"]):
                viscous_stress = stress.reshape((3, 3))
                press_tensor = np.identity(3) * press
                traction_tensor = viscous_stress + press_tensor
                traction = np.vstack((traction, traction_tensor @ normal))
                dev_force = np.vstack((dev_force, viscous_stress @ normal))
                press_force = np.vstack((press_force, press_tensor @ normal))

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
        total_traction = 0
        total_press_force = 0
        total_dev_force = 0

        for weight, traction, press, dev in zip(self.resampled_mesh.point_data["weights"],
                                                self.resampled_mesh.point_data["traction"],
                                                self.resampled_mesh.point_data["pressForce"],
                                                self.resampled_mesh.point_data["devForce"]):
            total_traction += weight * traction
            total_press_force += weight * press
            total_dev_force += weight * dev

        return total_traction, total_dev_force, total_press_force


# For testing purposes, using the main function
def main():
    processor = TractionAnalysis(
        "/home/data/analysis/Simulation/pressure_analysis/pressure_analysis_converted/z4/VTKFluid/Fluid_t200000.vtr",
        "/home/data/analysis/Simulation/pressure_analysis/pressure_analysis_converted/z4/VTKParticles/Particles_t200000.vtp")
    processor.process()
    traction, dev_force, total_press_force = processor.integrate_forces()
    print(f"traction -> {traction}\ndev -> {dev_force}\npress -> {total_press_force}\n")
    processor.save('test.vtp')


# The main function call remains the same
if __name__ == "__main__":
    main()

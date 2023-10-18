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
            self.fluid_data = pv.read(fluid_path)
            self.mesh_data = pv.read(mesh_path)
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
            cell_fluid = self.fluid_data
            self.resampled_mesh = self.mesh_data.sample(cell_fluid)
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


# For testing purposes, using the main function
def main():
    processor = TractionAnalysis("/home/data/analysis/Simulation/pressure_analysis/pressure_analysis_converted/z4/VTKFluid/Fluid_t200000.vtr",
                                 "/home/data/analysis/Simulation/pressure_analysis/pressure_analysis_converted/z4/VTKParticles/Particles_t200000.vtp")
    processor.process()
    processor.save('test.vtp')


# The main function call remains the same
if __name__ == "__main__":
    main()

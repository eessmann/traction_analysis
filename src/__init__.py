import pyvista as pv
import numpy as np


class TractionAnalysis:
    def __init__(self, fluid_path, mesh_path):
        self.resampled_mesh = None
        self.fluid_data = pv.read(fluid_path)
        self.mesh_data = pv.read(mesh_path)

    def fluid_interpolate(self):
        """Project Fluid data onto Particle"""
        cell_fluid = self.fluid_data
        self.resampled_mesh = self.mesh_data.sample(cell_fluid)
        return self

    def calculate_weights(self):
        """Calculate vertex weights for the mesh."""
        weights = np.zeros(self.resampled_mesh.n_points)
        for cell in self.resampled_mesh.cells:
            vec_a = cell.points[1] - cell.points[0]
            vec_b = cell.points[2] - cell.points[0]
            area = 0.5 * np.linalg.norm(np.cross(vec_a, vec_b))
            for vert in cell.point_ids:
                weights[vert] += area / 3.0
        self.resampled_mesh.point_data["weights"] = weights
        return self

    def calculate_traction(self):
        """Calculate traction force for the mesh."""
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
        return self

    def process(self):
        """Process the CFD data."""
        self.fluid_interpolate()
        self.calculate_weights()
        return self

    def save(self, output_path='processed_mesh.vtp'):
        """Save the processed mesh to a file."""
        self.resampled_mesh.save(output_path)


# For testing purposes, using the main function
def main():
    processor = TractionAnalysis("../data/raw/pressure_analysis/z4/VTKFluid/Fluid_t200000.vtr",
                                 "../data/raw/pressure_analysis/z4/VTKParticles/Particles_t200000.vtp")
    processor.process()
    processor.save('test.vtp')


# The main function call remains the same
if __name__ == "__main__":
    main()

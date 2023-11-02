import argparse as ap
import os
import re
from pathlib import Path

import tqdm as tm

from traction_analysis.analysis import particle_force_timeseries
from traction_analysis.file_handling import dir_path, is_empty_dir


# Recursively convert all fluid VTK file into new format
def convert_sim_dirs(input_path: str):
    sim_particle_pattern = re.compile(r"Particles_t(?P<timestep>\d+).vtp")
    sim_root = Path(input_path)

    def parse_sim_filename(pattern, key: str, file: str):
        res = pattern.search(file)
        return int(res.group(key))

    def conversion(path, files, pattern, conversion_func):
        # Parse the time steps of the simulation
        time_steps = sorted(list(set([parse_sim_filename(pattern, 'timestep', file) for file in files])))
        # Convert all fluid vtk files in VTKDirectory
        conversion_func(time_steps, path)

    for root, _, files in tm.tqdm(os.walk(input_path), desc="Walking Simulation Directory tree", position=0):
        path = Path(root)
        if (os.path.basename(path) == 'VTKParticles') and not is_empty_dir(path):
            conversion(path, files, sim_particle_pattern, particle_force_timeseries)


def main():
    """Main function to start the script."""
    parser = ap.ArgumentParser(prog="traction_analysis", description="Particle force analysis script")
    parser.add_argument("-i", "--input_path", type=dir_path, required=True,
                        help="Root directory of simulation campaign")
    args = parser.parse_args()
    convert_sim_dirs(args.input_path)

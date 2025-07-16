import numpy as np
import matplotlib.pyplot as plt
from potentials import ModelPotential


model_potential = ModelPotential(
    a=10000.0,
    b=100000.0,
    c=25000.0,
    make_asymmetric=True,
    time_dependent=True, # If False, ignore the rest of the parameters
    laser_amplitude=10000,
    laser_omega=3.0,
    laser_pulse_duration=0.4,
    laser_center_time=0.5,
    laser_envelope_type='gaussian',
    laser_spatial_profile_type='uniform',
    laser_charge=1.0,
    laser_polarization='linear_xy' # y: make only double well wiggle
)



print("\nCreating potential visualizations...")
model_potential.func.plot(
    n_points=100,
    plot_3d=True,
    save_path='figures/potential_animation_3d.gif',
    save_frames=True  # Set to True to save individual frames
)
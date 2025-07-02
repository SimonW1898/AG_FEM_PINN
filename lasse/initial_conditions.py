import dolfinx
import numpy as np
import matplotlib.pyplot as plt
from potential import double_well_potential, interaction_potential

x = np.linspace(-2.5, 2.5, 500)
v_dw = double_well_potential(x)
v_int_initial = interaction_potential(x, 0)
v_total_initial = v_dw + v_int_initial

plt.plot(x, v_total_initial)
plt.show()
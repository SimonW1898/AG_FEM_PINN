import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Potential Parameters ---
# Double well
a = 1.0
b = 5.0

# Dipole moment
q = 1.0  # Effective charge

# Laser pulse
E0 = 1.5      # E-field amplitude
omega = 2.25  # Carrier frequency
t0 = 20.0     # Pulse center
sigma_t = 5.0 # Pulse duration

def double_well_potential(x):
    """Calculate the double well potential V(x) = a*x^4 - b*x^2."""
    return a * x**4 - b * x**2

def dipole_moment(x):
    """Calculate the dipole moment mu(x) = q*x."""
    return q * x

def laser_pulse(t):
    """Calculate the time-dependent electric field of the laser pulse."""
    envelope = np.exp(-(t - t0)**2 / (2 * sigma_t**2))
    field = E0 * envelope * np.cos(omega * (t - t0))
    return field

def interaction_potential(x, t):
    """Calculate the interaction potential."""
    return -dipole_moment(x) * laser_pulse(t)

def total_potential(x, t):
    """Calculate the total time-dependent potential."""
    return double_well_potential(x) + interaction_potential(x, t)

# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(-2.5, 2.5, 500)
v_dw = double_well_potential(x)
v_int_initial = interaction_potential(x, 0)
v_total_initial = v_dw + v_int_initial

# Plot the lines
line_total, = ax.plot(x, v_total_initial, 'b-', linewidth=2, label='Total Potential V(x,t)')
line_dw, = ax.plot(x, v_dw, 'k--', linewidth=1.5, label='Double Well V(x)')
line_int, = ax.plot(x, v_int_initial, 'r:', linewidth=1.5, label='Interaction V_int(x,t)')

time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
ax.legend()

def init():
    """Initialize the animation."""
    ax.set_xlim(-2.5, 2.5)
    min_V = np.min(double_well_potential(x)) - abs(q * np.max(np.abs(x)) * E0)
    max_V = np.max(double_well_potential(x)) + abs(q * np.max(np.abs(x)) * E0)
    if max_V < 10: max_V = 10
    ax.set_ylim(min_V - 1, max_V)
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Potential Energy')
    ax.set_title('Time-Dependent Double Well Potential')
    ax.grid(True)
    
    line_total.set_ydata(v_total_initial)
    line_int.set_ydata(v_int_initial)
    
    time_text.set_text('')
    return line_total, line_int, time_text

def update(frame):
    """Update the plot for each frame of the animation."""
    t = frame * 0.1  # time
    
    v_int = interaction_potential(x, t)
    v_total = v_dw + v_int
    
    line_total.set_ydata(v_total)
    line_int.set_ydata(v_int)
    
    time_text.set_text(f'Time = {t:.2f}')
    return line_total, line_int, time_text

if __name__ == "__main__":

    # Create the animation
    # You may need to install ffmpeg to save the animation
    # On macOS: brew install ffmpeg
    # On Linux: sudo apt-get install ffmpeg
    ani = animation.FuncAnimation(fig, update, frames=range(400),
                                init_func=init, blit=True, interval=40)

    plt.show()

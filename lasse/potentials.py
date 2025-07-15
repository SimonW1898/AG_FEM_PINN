import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Callable, Optional
from abc import ABC, abstractmethod


class LaserPulse:
    """
    Laser pulse class for time-dependent potentials.
    
    Implements laser-dipole coupling in 2D:
    V_interaction(x, y, t) = -μ · E = -q*(x*Ex(t) + y*Ey(t))
    
    where μ = q*(x, y) is the dipole moment and E = (Ex(t), Ey(t)) is the electric field.
    """
    
    def __init__(self, amplitude: float, omega: float, pulse_duration: float, 
                 center_time: float = 0.5, phase: float = 0.0, 
                 envelope_type: str = 'gaussian', spatial_profile_type: str = 'uniform',
                 charge: float = 1.0, polarization: str = 'x'):
        """
        Parameters:
        - amplitude: Peak amplitude of the electric field
        - omega: Angular frequency of the laser
        - pulse_duration: Full width at half maximum (FWHM) of the pulse (normalized to [0,1])
        - center_time: Time at which the pulse is centered (normalized to [0,1])
        - phase: Phase of the oscillation
        - envelope_type: Type of temporal envelope ('gaussian', 'sech2', 'sin2')
        - spatial_profile_type: Spatial profile ('uniform', 'gaussian', 'x_linear', 'y_linear')
        - charge: Effective charge for dipole moment calculation
        - polarization: Electric field polarization ('x', 'y', 'circular', 'linear_xy')
        
        Note: All time parameters are normalized to [0,1]
        """
        self.amplitude = amplitude
        self.omega = omega
        self.pulse_duration = pulse_duration
        self.center_time = center_time
        self.phase = phase
        self.envelope_type = envelope_type
        self.spatial_profile_type = spatial_profile_type
        self.charge = charge
        self.polarization = polarization
    
    def temporal_envelope(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Calculate the temporal envelope of the pulse (normalized time t ∈ [0,1])."""
        t = np.asarray(t)
        tau = t - self.center_time  # Relative time from pulse center
        
        if self.envelope_type == 'gaussian':
            # Gaussian envelope: exp(-4*ln(2)*(t/T)^2) where T is FWHM
            sigma = self.pulse_duration / (2 * np.sqrt(2 * np.log(2)))
            return np.exp(-0.5 * (tau / sigma)**2)
        
        elif self.envelope_type == 'sech2':
            # Hyperbolic secant squared envelope
            return 1 / np.cosh(1.76 * tau / self.pulse_duration)**2
        
        elif self.envelope_type == 'sin2':
            # Sin^2 envelope for specified duration
            mask = np.abs(tau) <= self.pulse_duration / 2
            envelope = np.zeros_like(tau)
            envelope[mask] = np.sin(np.pi * tau[mask] / self.pulse_duration)**2
            return envelope
        
        else:
            raise ValueError(f"Unknown envelope type: {self.envelope_type}")
    
    def spatial_profile(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate the spatial profile of the pulse."""
        if self.spatial_profile_type == 'uniform':
            return np.ones_like(x)
        
        elif self.spatial_profile_type == 'gaussian':
            # Gaussian beam profile centered at (0.5, 0.5)
            r2 = (x - 0.5)**2 + (y - 0.5)**2
            return np.exp(-2 * r2 / 0.1)  # w0 = 0.1
        
        elif self.spatial_profile_type == 'x_linear':
            # Linear profile in x direction
            return x
        
        elif self.spatial_profile_type == 'y_linear':
            # Linear profile in y direction
            return y
        
        else:
            raise ValueError(f"Unknown spatial profile: {self.spatial_profile_type}")
    
    def electric_field(self, x: np.ndarray, y: np.ndarray, t: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the electric field components Ex(t) and Ey(t).
        
        Parameters:
        - x, y: Spatial coordinates
        - t: Time(s)
        
        Returns:
        - (Ex, Ey): Electric field components
        """
        temporal = self.temporal_envelope(t)
        spatial = self.spatial_profile(x, y)
        # Scale omega for normalized time t ∈ [0,1]: multiply by 2π to get proper frequency
        oscillation = np.cos(2 * np.pi * self.omega * np.asarray(t) + self.phase)
        
        field_magnitude = self.amplitude * temporal * spatial * oscillation
        
        if self.polarization == 'x':
            Ex = field_magnitude
            Ey = np.zeros_like(Ex)
        elif self.polarization == 'y':
            Ex = np.zeros_like(field_magnitude)
            Ey = field_magnitude
        elif self.polarization == 'linear_xy':
            # Linear polarization at 45 degrees
            Ex = field_magnitude / np.sqrt(2)
            Ey = field_magnitude / np.sqrt(2)
        elif self.polarization == 'circular':
            # Circular polarization
            Ex = field_magnitude * np.cos(2 * np.pi * self.omega * np.asarray(t) + self.phase)
            Ey = field_magnitude * np.sin(2 * np.pi * self.omega * np.asarray(t) + self.phase)
        else:
            raise ValueError(f"Unknown polarization: {self.polarization}")
        
        return Ex, Ey
    
    def evaluate(self, x: np.ndarray, y: np.ndarray, t: Union[float, np.ndarray]) -> np.ndarray:
        """
        Evaluate the laser-dipole interaction potential.
        
        Parameters:
        - x, y: Spatial coordinates
        - t: Time(s)
        
        Returns:
        - V_interaction = -μ · E = -q*(x*Ex + y*Ey)
        """
        Ex, Ey = self.electric_field(x, y, t)
        
        # Dipole moment components: μ = q*(x, y)
        mu_x = self.charge * (x - 0.5)
        mu_y = self.charge * (y - 0.5)
        
        # Interaction potential: V = -μ · E
        V_interaction = -(mu_x * Ex + mu_y * Ey)
        
        return V_interaction


class Potential(ABC):
    """
    Abstract base class for 2D potential functions.
    
    Provides a common interface for static and time-dependent potentials.
    Time-dependent potentials can include laser-dipole coupling effects.
    """
    
    def __init__(self, name: str, time_dependent: bool = False, 
                 laser_amplitude: float = 0.0, laser_omega: float = 1.0,
                 laser_pulse_duration: float = 0.3, laser_center_time: float = 0.5,
                 laser_phase: float = 0.0, laser_envelope_type: str = 'gaussian',
                 laser_spatial_profile_type: str = 'uniform', laser_charge: float = 1.0,
                 laser_polarization: str = 'x'):
        self.name = name
        self.time_dependent = time_dependent
        
        # Create laser pulse if time-dependent
        if time_dependent:
            self.laser_pulse = LaserPulse(
                amplitude=laser_amplitude,
                omega=laser_omega,
                pulse_duration=laser_pulse_duration,
                center_time=laser_center_time,
                phase=laser_phase,
                envelope_type=laser_envelope_type,
                spatial_profile_type=laser_spatial_profile_type,
                charge=laser_charge,
                polarization=laser_polarization
            )
        else:
            self.laser_pulse = None
    
    @abstractmethod
    def _potential_function(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Abstract method to define the 2D potential function.
        Must be implemented by subclasses.
        
        Parameters:
        - x, y: Position arrays
        - **kwargs: Additional parameters for the potential
        
        Returns:
        - Potential values
        """
        pass
    
    def get_array(self, n_points: int = 100, x_range: Tuple[float, float] = (0, 1),
                  y_range: Tuple[float, float] = (0, 1), **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 2D potential as arrays.
        
        Parameters:
        - n_points: Number of grid points in each direction
        - x_range: (x_min, x_max) range for evaluation
        - y_range: (y_min, y_max) range for evaluation
        - **kwargs: Additional parameters for the potential
        
        Returns:
        - (X, Y, V) where X, Y are meshgrids and V is the potential
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        V = self._potential_function(X, Y, **kwargs)
        return X, Y, V
    
    def plot(self, n_points: int = 100, time_range: Tuple[float, float] = (0, 1), 
             n_frames: int = 50, plot_3d: bool = False, **kwargs):
        """
        Plot 2D potential.
        
        Parameters:
        - n_points: Number of spatial grid points
        - time_range: (t_min, t_max) for animation if time_dependent=True
        - n_frames: Number of frames for animation
        - plot_3d: If True, create 3D surface plot (2D contour if False)
        - **kwargs: Additional parameters for the potential
        """
        if self.time_dependent:
            if plot_3d:
                self._plot_animated_3d(n_points, time_range, n_frames, **kwargs)
            else:
                self._plot_animated(n_points, time_range, n_frames, **kwargs)
        else:
            self._plot_static(n_points, **kwargs)
    
    def _plot_static(self, n_points: int, **kwargs):
        """Plot static 2D potential."""
        X, Y, V = self.get_array(n_points, **kwargs)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Contour plot
        contour = ax1.contourf(X, Y, V, levels=20, cmap='viridis')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'{self.name} Potential (2D Contour)')
        plt.colorbar(contour, ax=ax1)
        
        # 3D surface plot
        ax2 = fig.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(X, Y, V, cmap='viridis', alpha=0.8)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('V(x,y)')
        ax2.set_title(f'{self.name} Potential (3D Surface)')
        plt.colorbar(surf, ax=ax2, shrink=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_animated(self, n_points: int, time_range: Tuple[float, float], 
                       n_frames: int, **kwargs):
        """Plot animated 2D potential."""
        from matplotlib.animation import FuncAnimation
        
        x = np.linspace(0, 1, n_points)
        y = np.linspace(0, 1, n_points)
        X, Y = np.meshgrid(x, y)
        times = np.linspace(time_range[0], time_range[1], n_frames)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate V range for consistent color limits
        V_all = []
        for t in times:
            V_all.extend(self.evaluate_time_dependent(X, Y, t, **kwargs).flatten())
        vmin, vmax = min(V_all), max(V_all)
        
        # Initial plot
        V0 = self.evaluate_time_dependent(X, Y, times[0], **kwargs)
        im = ax.contourf(X, Y, V0, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{self.name} Potential (2D, t={times[0]:.2f})')
        plt.colorbar(im, ax=ax)
        
        def animate(frame):
            ax.clear()
            t = times[frame]
            V = self.evaluate_time_dependent(X, Y, t, **kwargs)
            im = ax.contourf(X, Y, V, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'{self.name} Potential (2D, t={t:.2f})')
            # No need to return anything when blit=False
        
        anim = FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=True, blit=False)
        plt.show()
        return anim
    
    def _plot_animated_3d(self, n_points: int, time_range: Tuple[float, float], 
                          n_frames: int, **kwargs):
        """Plot animated 3D surface potential."""
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d import Axes3D
        
        x = np.linspace(0, 1, n_points)
        y = np.linspace(0, 1, n_points)
        X, Y = np.meshgrid(x, y)
        times = np.linspace(time_range[0], time_range[1], n_frames)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate V range for consistent z-axis limits
        V_all = []
        for t in times:
            V_all.extend(self.evaluate_time_dependent(X, Y, t, **kwargs).flatten())
        vmin, vmax = min(V_all), max(V_all)
        
        # Initial surface plot
        V0 = self.evaluate_time_dependent(X, Y, times[0], **kwargs)
        surf = ax.plot_surface(X, Y, V0, cmap='viridis', alpha=0.8, 
                              vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('V(x,y,t)')
        ax.set_title(f'{self.name} Potential (3D, t={times[0]:.2f})')
        ax.set_zlim(vmin, vmax)
        
        # Add colorbar
        cbar = plt.colorbar(surf, ax=ax, shrink=0.5)
        
        def animate(frame):
            ax.clear()
            t = times[frame]
            V = self.evaluate_time_dependent(X, Y, t, **kwargs)
            
            surf = ax.plot_surface(X, Y, V, cmap='viridis', alpha=0.8,
                                  vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('V(x,y,t)')
            ax.set_title(f'{self.name} Potential (3D, t={t:.2f})')
            ax.set_zlim(vmin, vmax)
            
            # Keep the same view angle
            ax.view_init(elev=30, azim=45)
            
            return surf,
        
        anim = FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=True, blit=False)
        plt.show()
        return anim
    
    def evaluate_time_dependent(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray], 
                               t: Union[float, np.ndarray], **kwargs) -> np.ndarray:
        """
        Evaluate time-dependent potential in 2D.
        
        Parameters:
        - x, y: Position arrays
        - t: Time(s)
        - **kwargs: Additional parameters for the potential
        
        Returns:
        - V(x, y, t) = V_static(x, y) + V_laser(x, y, t)
        """
        if not self.time_dependent:
            raise ValueError("This potential is not time-dependent")
        
        # Static potential
        V_static = self._potential_function(x, y, **kwargs)
        
        # Laser pulse contribution
        V_laser = self.laser_pulse.evaluate(x, y, t)
        
        return V_static + V_laser
    
    def __call__(self, *args, **kwargs):
        """
        Make the potential callable like a function.
        
        Parameters:
        - For static 2D: __call__(x, y, **kwargs)
        - For time-dependent 2D: __call__(x, y, t, **kwargs)
        
        Returns:
        - Potential value(s)
        """
        if self.time_dependent:
            if len(args) == 3:
                # Time-dependent 2D case: x, y, t
                return self.evaluate_time_dependent(args[0], args[1], args[2], **kwargs)
            else:
                raise ValueError("Time-dependent potentials expect 3 args (x, y, t)")
        else:
            if len(args) == 2:
                # Static 2D case
                return self._potential_function(args[0], args[1], **kwargs)
            else:
                raise ValueError("Static potentials expect 2 args (x, y)")


class HarmonicPotential(Potential):
    """
    2D Harmonic potential: V(x,y) = depth * 2 * ((x - 0.5)^2 + (y - 0.5)^2 - 0.5)
    
    This potential is zero at the corners (0,0), (0,1), (1,0), (1,1) and has
    its minimum at the center (0.5, 0.5) with value -depth.
    """
    
    def __init__(self, time_dependent: bool = False, **kwargs):
        super().__init__("Harmonic", time_dependent=time_dependent, **kwargs)
    
    def _potential_function(self, x: np.ndarray, y: np.ndarray, depth: float = 1.0) -> np.ndarray:
        """
        Harmonic potential function.
        
        Parameters:
        - x, y: Position arrays in [0,1] x [0,1]
        - depth: Depth of the potential well (positive value)
        
        Returns:
        - V(x,y) = depth * 2 * ((x - 0.5)^2 + (y - 0.5)^2 - 0.5)
        """
        return depth * 2 * ((x - 0.5)**2 + (y - 0.5)**2 - 0.5)


class DoubleWell(Potential):
    """
    2D Double well potential: V(x,y) = depth * 32 * (f(x) + f(y))
    where f(t) = (t - 0.5)^4 - 0.25 * (t - 0.5)^2
    
    This potential is zero at the corners and center, with four wells
    located symmetrically around the center.
    """
    
    def __init__(self, time_dependent: bool = False, **kwargs):
        super().__init__("Double Well", time_dependent=time_dependent, **kwargs)
    
    def _potential_function(self, x: np.ndarray, y: np.ndarray, depth: float = 1.0) -> np.ndarray:
        """
        Double well potential function.
        
        Parameters:
        - x, y: Position arrays in [0,1] x [0,1]
        - depth: Depth of the potential wells (positive value)
        
        Returns:
        - V(x,y) = depth * 32 * (f(x) + f(y))
        - where f(t) = (t - 0.5)^4 - 0.25 * (t - 0.5)^2
        """
        def f(t):
            return (t - 0.5)**4 - 0.25 * (t - 0.5)**2
        
        return depth * 32 * (f(x) + f(y))


class ModelPotential(Potential):
    """
    Model potential formed by combining 1D harmonic in x-direction and 1D double well in y-direction.
    V(x,y) = V_harmonic(x) + V_double_well(y)
    
    This creates a 2D potential with harmonic confinement in x and double well structure in y.
    """
    
    def __init__(self, time_dependent: bool = False, **kwargs):
        super().__init__("Model (Harmonic x + Double Well y)", time_dependent=time_dependent, **kwargs)
    
    def _potential_function(self, x: np.ndarray, y: np.ndarray, x_depth: float = 1.0, y_depth: float = 1.0) -> np.ndarray:
        """
        Model potential function combining 1D harmonic and 1D double well.
        
        Parameters:
        - x, y: Position arrays in [0,1] x [0,1]
        - x_depth: Depth of the harmonic potential in x-direction
        - y_depth: Depth of the double well potential in y-direction
        
        Returns:
        - V(x,y) = V_harmonic(x) + V_double_well(y)
        - where V_harmonic(x) = x_depth * 4 * ((x - 0.5)^2 - 0.25)
        - and V_double_well(y) = y_depth * 64 * ((y - 0.5)^4 - 0.25 * (y - 0.5)^2)
        """
        # 1D harmonic in x-direction (zero at x=0,1, minimum at x=0.5)
        V_harmonic_x = x_depth * 4 * ((x - 0.5)**2 - 0.25)
        
        # 1D double well in y-direction (zero at y=0,0.5,1, minima at y≈0.146,0.854)
        V_double_well_y = y_depth * 64 * ((y - 0.5)**4 - 0.25 * (y - 0.5)**2)
        
        return V_harmonic_x + V_double_well_y


# Example usage and testing
if __name__ == "__main__":
    # Test Time-Dependent Model Potential
    V = ModelPotential(
        time_dependent=True,
        laser_amplitude=0.4,
        laser_omega=5.0,
        laser_pulse_duration=0.4,
        laser_center_time=0.5,
        laser_envelope_type='gaussian',
        laser_spatial_profile_type='uniform',
        laser_charge=1.0,
        laser_polarization='linear_xy'
    )
    
    # Evaluate potential with 2d meshgrid at constant time
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    print(V(X, Y, 0.5).shape)

    # Plot time-dependent model potential animation (3D)
    print("Plotting Time-Dependent Model Potential Animation (3D)...")
    V.plot(time_range=(0, 1), n_frames=30, plot_3d=True, x_depth=1.0, y_depth=1.0)
    
    


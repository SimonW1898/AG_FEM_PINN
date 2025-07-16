import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import os
from typing import Union, Tuple, Callable, Optional
from abc import ABC, abstractmethod

# Global plotting configuration
POTENTIAL_CMAP = 'viridis'
POTENTIAL_ALPHA = 0.8

# Font size configuration for consistency
PLOT_TITLE_FONTSIZE = 14
PLOT_LABEL_FONTSIZE = 12
PLOT_TICK_FONTSIZE = 10

# Configure matplotlib for consistent font sizes
plt.rcParams.update({
    'font.size': PLOT_TICK_FONTSIZE,
    'axes.titlesize': PLOT_TITLE_FONTSIZE,
    'axes.labelsize': PLOT_LABEL_FONTSIZE,
    'xtick.labelsize': PLOT_TICK_FONTSIZE,
    'ytick.labelsize': PLOT_TICK_FONTSIZE,
    'legend.fontsize': PLOT_TICK_FONTSIZE
})


class LaserPulse:
    """
    Laser pulse class for time-dependent potentials.
    
    Implements laser-dipole coupling in N dimensions:
    V_interaction(x, t) = -μ · E = -q*sum(x_i*E_i(t))
    
    where μ = q*x is the dipole moment and E(t) is the electric field.
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
        - spatial_profile_type: Spatial profile ('uniform', 'gaussian', 'linear')
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
    
    def spatial_profile(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the spatial profile of the pulse.
        
        Parameters:
        - x: Position array of shape (dim, N)
        """
        if self.spatial_profile_type == 'uniform':
            return np.ones(x.shape[1])
        
        elif self.spatial_profile_type == 'gaussian':
            # Gaussian beam profile centered at 0.5 in each dimension
            r2 = np.sum((x - 0.5)**2, axis=0)
            return np.exp(-2 * r2 / 0.1)  # w0 = 0.1
        
        elif self.spatial_profile_type == 'linear':
            # Linear profile in first dimension
            return x[0]
        
        else:
            raise ValueError(f"Unknown spatial profile: {self.spatial_profile_type}")
    
    def electric_field(self, x: np.ndarray, t: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate the electric field components E(t).
        
        Parameters:
        - x: Position array of shape (dim, N)
        - t: Time(s)
        
        Returns:
        - E: Electric field components of shape (dim, N)
        """
        temporal = self.temporal_envelope(t)
        spatial = self.spatial_profile(x)
        # Scale omega for normalized time t ∈ [0,1]: multiply by 2π to get proper frequency
        oscillation = np.cos(2 * np.pi * self.omega * np.asarray(t) + self.phase)
        
        field_magnitude = self.amplitude * temporal * spatial * oscillation
        dim = x.shape[0]
        
        if self.polarization == 'x':
            E = np.zeros((dim, x.shape[1]))
            E[0] = field_magnitude
        elif self.polarization == 'y' and dim >= 2:
            E = np.zeros((dim, x.shape[1]))
            E[1] = field_magnitude
        elif self.polarization == 'linear_xy' and dim >= 2:
            # Linear polarization at 45 degrees in x-y plane
            E = np.zeros((dim, x.shape[1]))
            E[0] = field_magnitude / np.sqrt(2)
            E[1] = field_magnitude / np.sqrt(2)
        elif self.polarization == 'circular' and dim >= 2:
            # Circular polarization in x-y plane
            E = np.zeros((dim, x.shape[1]))
            E[0] = field_magnitude * np.cos(2 * np.pi * self.omega * np.asarray(t) + self.phase)
            E[1] = field_magnitude * np.sin(2 * np.pi * self.omega * np.asarray(t) + self.phase)
        else:
            raise ValueError(f"Unknown or incompatible polarization: {self.polarization} for dimension {dim}")
        
        return E
    
    def evaluate(self, x: np.ndarray, t: Union[float, np.ndarray]) -> np.ndarray:
        """
        Evaluate the laser-dipole interaction potential.
        
        Parameters:
        - x: Position array of shape (dim, N)
        - t: Time(s)
        
        Returns:
        - V_interaction = -μ · E = -q*sum(x_i*E_i)
        """
        E = self.electric_field(x, t)
        
        # Dipole moment: μ = q*(x - 0.5)
        mu = self.charge * (x - 0.5)
        
        # Interaction potential: V = -μ · E
        V_interaction = -np.sum(mu * E, axis=0)
        
        return V_interaction


class Potential(ABC):
    """
    Abstract base class for N-dimensional potential functions.
    
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
    def _potential_function(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Abstract method to define the N-dimensional potential function.
        Must be implemented by subclasses.
        
        Parameters:
        - x: Position array of shape (dim, N)
        - **kwargs: Additional parameters for the potential
        
        Returns:
        - Potential values of shape (N,)
        """
        pass
    
    def get_array(self, n_points: int = 100, ranges: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get N-dimensional potential as arrays.
        
        Parameters:
        - n_points: Number of grid points in each direction
        - ranges: Array of shape (dim, 2) specifying (min, max) for each dimension
                 If None, uses (0,1) for all dimensions
        - **kwargs: Additional parameters for the potential
        
        Returns:
        - (X, V) where X has shape (dim, n_points^dim) and V has shape (n_points^dim,)
        """
        # Default range is [0,1] in each dimension
        if ranges is None:
            ranges = np.array([[0, 1], [0, 1]])
        
        # Create meshgrid
        dim = ranges.shape[0]
        grids = [np.linspace(ranges[i,0], ranges[i,1], n_points) for i in range(dim)]
        X = np.meshgrid(*grids, indexing='ij')
        
        # Reshape to (dim, N) array
        X = np.array([x.flatten() for x in X])
        
        # Evaluate potential
        V = self._potential_function(X, **kwargs)
        
        return X, V
    
    def plot(self, n_points: int = 100, time_range: Tuple[float, float] = (0, 1), 
             n_frames: int = 50, plot_3d: bool = False, save_path: Optional[str] = None, 
             save_frames: bool = False, total_duration: float = 5.0, **kwargs):
        """
        Plot potential (only implemented for 2D potentials).
        
        Parameters:
        - n_points: Number of spatial grid points
        - time_range: (t_min, t_max) for animation if time_dependent=True
        - n_frames: Number of frames for animation
        - plot_3d: If True, create 3D surface plot (2D contour if False)
        - save_path: Optional path to save the plot/animation
        - save_frames: If True, save individual frames to figures/potential_frames/
        - total_duration: Total duration of the animation in seconds (default: 5.0)
        - **kwargs: Additional parameters for the potential
        """
        if self.time_dependent:
            if plot_3d:
                self._plot_animated_3d(n_points, time_range, n_frames, save_path, save_frames, total_duration, **kwargs)
            else:
                self._plot_animated(n_points, time_range, n_frames, save_path, save_frames, **kwargs)
        else:
            self._plot_static(n_points, save_path, **kwargs)
    
    def _plot_static(self, n_points: int, save_path: Optional[str] = None, **kwargs):
        """Plot static 2D potential."""
        X, V = self.get_array(n_points, **kwargs)
        if X.shape[0] != 2:
            raise ValueError("Plotting only implemented for 2D potentials")
        
        # Reshape back to 2D grid
        X1 = X[0].reshape(n_points, n_points)
        X2 = X[1].reshape(n_points, n_points)
        V = V.reshape(n_points, n_points)
        
        fig = plt.figure(figsize=(12, 5))
        
        # Contour plot
        ax1 = fig.add_subplot(121)
        contour = ax1.contourf(X1, X2, V, levels=20, cmap=POTENTIAL_CMAP)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'{self.name} Potential (2D Contour)')
        plt.colorbar(contour, ax=ax1)
        
        # 3D surface plot
        ax2 = fig.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(X1, X2, V, cmap=POTENTIAL_CMAP, alpha=POTENTIAL_ALPHA)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('V(x,y)')
        ax2.set_title(f'{self.name} Potential (3D Surface)')
        ax2.view_init(elev=30, azim=45)
        plt.colorbar(surf, ax=ax2, shrink=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
            print(f"Static plot saved to: {save_path}")
        
        else:
            plt.show()
    
    def _plot_animated(self, n_points: int, time_range: Tuple[float, float], 
                       n_frames: int, save_path: Optional[str] = None, save_frames: bool = False, **kwargs):
        """Plot animated 2D potential."""
        from matplotlib.animation import FuncAnimation
        
        # Create frame directory if saving frames
        if save_frames:
            frame_dir = 'figures/potential_frames'
            os.makedirs(frame_dir, exist_ok=True)
        
        # Create coordinate grid using the same method as get_array
        ranges = np.array([[0, 1], [0, 1]])  # Default range
        grids = [np.linspace(ranges[i,0], ranges[i,1], n_points) for i in range(2)]
        X_mesh = np.meshgrid(*grids, indexing='ij')
        
        # Reshape to (dim, N) array for evaluation
        X = np.array([x.flatten() for x in X_mesh])
        
        # Reshape back to 2D grid for plotting
        X1 = X_mesh[0]  # Already in correct shape (n_points, n_points)
        X2 = X_mesh[1]  # Already in correct shape (n_points, n_points)
        
        times = np.linspace(time_range[0], time_range[1], n_frames)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate V range for consistent color limits
        V_all = []
        for t in times:
            V_all.extend(self.evaluate_time_dependent(X, t, **kwargs).flatten())
        vmin, vmax = min(V_all), max(V_all)
        
        # Initial plot
        V0 = self.evaluate_time_dependent(X, times[0], **kwargs)
        V0 = V0.reshape(n_points, n_points)
        im = ax.contourf(X1, X2, V0, levels=20, cmap=POTENTIAL_CMAP, vmin=vmin, vmax=vmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{self.name} Potential (2D, t={times[0]:.2f})')
        plt.colorbar(im, ax=ax)
        
        # Save initial frame if requested
        if save_frames:
            frame_path = os.path.join(frame_dir, f'potential_frame_{0:04d}.png')
            plt.savefig(frame_path, dpi=300, bbox_inches='tight')
        
        def animate(frame):
            ax.clear()
            t = times[frame]
            V = self.evaluate_time_dependent(X, t, **kwargs)
            V = V.reshape(n_points, n_points)
            im = ax.contourf(X1, X2, V, levels=20, cmap=POTENTIAL_CMAP, vmin=vmin, vmax=vmax)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'{self.name} Potential (2D, t={t:.2f})')
            
            # Save frame if requested
            if save_frames:
                frame_path = os.path.join(frame_dir, f'potential_frame_{frame:04d}.png')
                plt.savefig(frame_path, dpi=300, bbox_inches='tight')
            
            return [im]
        
        anim = FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=True, blit=True)
        
        if save_path:
            # Save animation as GIF
            anim.save(save_path, writer='pillow', fps=5)
            print(f"2D animation saved to: {save_path}")
            plt.close()
        else:
            plt.show()
        
        if save_frames:
            print(f"2D potential frames saved to: {frame_dir}")
        
        return anim
    
    def _plot_animated_3d(self, n_points: int, time_range: Tuple[float, float], 
                          n_frames: int, save_path: Optional[str] = None, save_frames: bool = False, 
                          total_duration: float = 5.0, **kwargs):
        """Plot animated 3D surface potential."""
        from matplotlib.animation import FuncAnimation
        
        # Create coordinate grid using the same method as get_array
        ranges = np.array([[0, 1], [0, 1]])  # Default range
        grids = [np.linspace(ranges[i,0], ranges[i,1], n_points) for i in range(2)]
        X_mesh = np.meshgrid(*grids, indexing='ij')
        
        # Reshape to (dim, N) array for evaluation
        X = np.array([x.flatten() for x in X_mesh])
        
        # Get coordinates for plotting (flattened for plot_trisurf)
        X_coords = X[0]  # x coordinates
        Y_coords = X[1]  # y coordinates
        
        times = np.linspace(time_range[0], time_range[1], n_frames)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate frame rate to achieve desired total duration
        fps = n_frames / total_duration
        interval_ms = 1000 / fps  # Convert to milliseconds for matplotlib
        
        # Calculate V range for consistent z-axis limits
        V_all = []
        for t in times:
            V_all.extend(self.evaluate_time_dependent(X, t, **kwargs).flatten())
        vmin, vmax = min(V_all), max(V_all)
        
        print(f"3D Animation: V range [{vmin:.4f}, {vmax:.4f}]")
        print(f"3D Animation: Using {len(X_coords)} points")
        
        # Initial surface plot using plot_trisurf with consistent settings
        V0 = self.evaluate_time_dependent(X, times[0], **kwargs)
        print(f"3D Animation: Initial V0 shape: {V0.shape}, min: {V0.min():.4f}, max: {V0.max():.4f}")
        
        surf = ax.plot_trisurf(X_coords, Y_coords, V0, cmap=POTENTIAL_CMAP, alpha=POTENTIAL_ALPHA, 
                              linewidth=0, antialiased=True)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('V(x,y,t)')
        ax.set_title(f'{self.name} Potential (t = {times[0]:.4f})')
        ax.set_zlim(vmin, vmax)
        
        # Set consistent view angle
        ax.view_init(elev=30, azim=45)
        
        # Create frame directory if saving frames
        if save_frames:
            frame_dir = 'figures/potential_frames'
            os.makedirs(frame_dir, exist_ok=True)
        
        def animate(frame):
            ax.clear()
            t = times[frame]
            V = self.evaluate_time_dependent(X, t, **kwargs)
            
            surf = ax.plot_trisurf(X_coords, Y_coords, V, cmap=POTENTIAL_CMAP, alpha=POTENTIAL_ALPHA,
                                  linewidth=0, antialiased=True)
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('V(x,y,t)')
            ax.set_title(f'{self.name} Potential (t = {t:.4f})')
            ax.set_zlim(vmin, vmax)
            
            # Keep the same view angle
            ax.view_init(elev=30, azim=45)
            
            # Apply consistent layout for animation frames
            ax.set_box_aspect(None, zoom=0.85)
            
            # Save frame if requested
            if save_frames:
                frame_path = os.path.join(frame_dir, f'potential_frame_{frame:04d}.png')
                plt.savefig(frame_path, dpi=300, bbox_inches='tight')
            
            return [surf]
        
        anim = FuncAnimation(fig, animate, frames=n_frames, interval=interval_ms, repeat=True, blit=True)
        
        if save_path:
            # Save animation as GIF
            anim.save(save_path, writer='pillow', fps=int(fps))
            print(f"3D animation saved to: {save_path} ({n_frames} frames, {fps:.1f} fps, {total_duration}s duration)")
            plt.close()
        else:
            plt.show()
        
        if save_frames:
            print(f"3D potential frames saved to: {frame_dir}")
        
        return anim
    
    def evaluate_time_dependent(self, x: np.ndarray, t: Union[float, np.ndarray], **kwargs) -> np.ndarray:
        """
        Evaluate time-dependent potential.
        
        Parameters:
        - x: Position array of shape (dim, N)
        - t: Time(s)
        - **kwargs: Additional parameters for the potential
        
        Returns:
        - V(x, t) = V_static(x) + V_laser(x, t)
        """
        if not self.time_dependent:
            raise ValueError("This potential is not time-dependent")
        
        # Static potential
        V_static = self._potential_function(x, **kwargs)
        
        # Laser pulse contribution
        if self.laser_pulse is not None:
            V_laser = self.laser_pulse.evaluate(x, t)
            return V_static + V_laser
        else:
            return V_static
    
    def __call__(self, *args, **kwargs):
        """
        Make the potential callable like a function.
        
        Parameters:
        - For static: __call__(x, **kwargs) where x has shape (dim, N)
        - For time-dependent: __call__(x, t, **kwargs)
        
        Returns:
        - Potential value(s)
        """
        if self.time_dependent:
            if len(args) == 2:
                # Time-dependent case: x, t
                return self.evaluate_time_dependent(args[0], args[1], **kwargs)
            else:
                raise ValueError("Time-dependent potentials expect 2 args (x, t)")
        else:
            if len(args) == 1:
                # Static case
                return self._potential_function(args[0], **kwargs)
            else:
                raise ValueError("Static potentials expect 1 arg (x)")


class HarmonicPotential(Potential):
    """
    2D Harmonic potential: V(x) = depth * 2 * (sum((x_i - 0.5)^2) - 0.5)
    
    This potential is zero at the corners and has its minimum at the center (0.5, 0.5)
    with value -depth.
    """
    
    def __init__(self, time_dependent: bool = False, **kwargs):
        super().__init__("Harmonic", time_dependent=time_dependent, **kwargs)
    
    def _potential_function(self, x: np.ndarray, depth: float = 1.0) -> np.ndarray:
        """
        Harmonic potential function.
        
        Parameters:
        - x: Position array of shape (dim, N)
        - depth: Depth of the potential well (positive value)
        
        Returns:
        - V(x) = depth * 2 * (sum((x_i - 0.5)^2) - 0.5)
        """
        return depth * 2 * (np.sum((x - 0.5)**2, axis=0) - 0.5)


class DoubleWell(Potential):
    """
    2D Double well potential: V(x) = depth * 32 * sum(f(x_i))
    where f(t) = (t - 0.5)^4 - 0.25 * (t - 0.5)^2
    
    This potential is zero at the corners and center, with four wells
    located symmetrically around the center.
    """
    
    def __init__(self, time_dependent: bool = False, **kwargs):
        super().__init__("Double Well", time_dependent=time_dependent, **kwargs)
    
    def _potential_function(self, x: np.ndarray, depth: float = 1.0) -> np.ndarray:
        """
        Double well potential function.
        
        Parameters:
        - x: Position array of shape (dim, N)
        - depth: Depth of the potential wells (positive value)
        
        Returns:
        - V(x) = depth * 32 * sum(f(x_i))
        - where f(t) = (t - 0.5)^4 - 0.25 * (t - 0.5)^2
        """
        def f(t):
            return (t - 0.5)**4 - 0.25 * (t - 0.5)**2
        
        return depth * 32 * np.sum(f(x), axis=0)


class ModelPotentialOld(Potential):
    """
    Model potential formed by combining 1D harmonic in x-direction and 1D double well in y-direction.
    V(x) = V_harmonic(x_1) + V_double_well(x_2)
    
    This creates a 2D potential with harmonic confinement in x and double well structure in y.
    """
    
    def __init__(self, 
                 time_dependent: bool = False, 
                 x_depth: float = 1.0, 
                 y_depth: float = 1.0,
                 make_asymmetric: bool = False,
                 **kwargs):
        self.x_depth = x_depth
        self.y_depth = y_depth
        self.make_asymmetric = make_asymmetric
        super().__init__("Model (Harmonic x + Double Well y)", time_dependent=time_dependent, **kwargs)
    
    def _potential_function(self, x: np.ndarray) -> np.ndarray:
        """
        Model potential function combining 1D harmonic and 1D double well.
        
        Parameters:
        - x: Position array of shape (dim, N)
        - x_depth: Depth of the harmonic potential in x-direction
        - y_depth: Depth of the double well potential in y-direction
        
        Returns:
        - V(x) = V_harmonic(x_1) + V_double_well(x_2)
        - where V_harmonic(x) = x_depth * 4 * ((x - 0.5)^2 - 0.25)
        - and V_double_well(y) = y_depth * 64 * ((y - 0.5)^4 - 0.25 * (y - 0.5)^2)
        """
        # 1D harmonic in x-direction (zero at x=0,1, minimum at x=0.5)
        V_harmonic_x = self.x_depth * 4 * ((x[0] - 0.5)**2 - 0.25)

        if self.make_asymmetric:
            V_harmonic_x += 0.01 * (x[0] - 0.5)
        
        # 1D double well in y-direction (zero at y=0,0.5,1, minima at y≈0.146,0.854)
        V_double_well_y = self.y_depth * 64 * ((x[1] - 0.5)**4 - 0.25 * (x[1] - 0.5)**2)

        if self.make_asymmetric:
            V_double_well_y += 0.01 * (x[1] - 0.5)
        
        return V_harmonic_x + V_double_well_y + self.x_depth + self.y_depth

class ModelPotential(Potential):
    """
    Model potential formed by combining 1D harmonic in x-direction and 1D double well in y-direction.
    V(x) = V_harmonic(x_1) + V_double_well(x_2)
    
    This creates a 2D potential with harmonic confinement in x and double well structure in y.
    """
    
    def __init__(self, 
                 time_dependent: bool = False, 
                 a: float = 1.0, 
                 b: float = 1.0,
                 c: float = 1.0,
                 make_asymmetric: bool = False,
                 **kwargs):
        self.a = a
        self.b = b
        self.c = c
        self.make_asymmetric = make_asymmetric
        super().__init__("Model (Harmonic x + Double Well y)", time_dependent=time_dependent, **kwargs)
    
    def _potential_function(self, x: np.ndarray) -> np.ndarray:
        """
        Model potential function combining 1D harmonic and 1D double well.
        
        Parameters:
        - x: Position array of shape (dim, N)
        
        Returns:
        - V(x) = V_harmonic(x_1) + V_double_well(x_2)
        - where V_harmonic(x) = a * (x - 0.5)^2
        - and V_double_well(y) = b * (y - 0.5)^4 - c * (y - 0.5)^2
        """
        # 1D harmonic in x-direction (zero at x=0,1, minimum at x=0.5)
        V_harmonic_x = self.a * (x[0] - 0.5)**2

        if self.make_asymmetric:
            V_harmonic_x += 0.01 * (x[0] - 0.5)
        
        # 1D double well in y-direction (zero at y=0,0.5,1, minima at y≈0.146,0.854)
        V_double_well_y = self.b * (x[1] - 0.5)**4 - self.c * (x[1] - 0.5)**2

        if self.make_asymmetric:
            V_double_well_y += 0.01 * (x[1] - 0.5)
        
        V_total = V_harmonic_x + V_double_well_y
        return V_total - np.min(V_total)


# Example usage and testing
if __name__ == "__main__":
    # Test Time-Dependent Model Potential
    V = ModelPotential(
        x_depth=100.0,
        y_depth=1000.0,
        make_asymmetric=True,
        time_dependent=True,
        laser_amplitude=100,
        laser_omega=3.0,
        laser_pulse_duration=0.4,
        laser_center_time=0.5,
        laser_envelope_type='gaussian',
        laser_spatial_profile_type='uniform',
        laser_charge=1.0,
        laser_polarization='linear_xy'
    )
    
    # Create coordinate grid
    n_points = 100
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)
    coords = np.array([X.flatten(), Y.flatten()])  # Shape: (2, n_points^2)
    
    # Evaluate potential at constant time
    print(V(coords, 0.5).shape)  # Should print: (10000,)
    print(V(coords, 0.5))

    # Plot time-dependent model potential animation (3D)
    print("Plotting Time-Dependent Model Potential Animation (3D)...")
    V.plot(time_range=(0, 1), n_frames=30, plot_3d=True, save_path='figures/model_potential_3d_animation.gif')
    
    


import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Callable
from abc import ABC, abstractmethod


class Potential(ABC):
    """
    Abstract base class for potential functions.
    All potentials are defined on the domain [0,1] for both 1D and 2D.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def _potential_function(self, *coords) -> float:
        """
        Abstract method to define the potential function.
        Must be implemented by subclasses.
        """
        pass
    
    def evaluate_1d(self, x: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
        """
        Evaluate potential in 1D.
        
        Parameters:
        - x: Position(s) in [0,1]
        - **kwargs: Additional parameters for the potential
        
        Returns:
        - Potential value(s)
        """
        # Ensure x is in [0,1]
        x = np.asarray(x)
        if np.any((x < 0) | (x > 1)):
            raise ValueError("x coordinates must be in [0,1]")
        
        return self._potential_function(x, **kwargs)
    
    def evaluate_2d(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray], **kwargs) -> np.ndarray:
        """
        Evaluate potential in 2D.
        
        Parameters:
        - x, y: Position(s) in [0,1] x [0,1]
        - **kwargs: Additional parameters for the potential
        
        Returns:
        - Potential value(s)
        """
        # Ensure coordinates are in [0,1]
        x, y = np.asarray(x), np.asarray(y)
        if np.any((x < 0) | (x > 1)) or np.any((y < 0) | (y > 1)):
            raise ValueError("x,y coordinates must be in [0,1]")
        
        return self._potential_function(x, y, **kwargs)
    
    def get_1d_array(self, n_points: int = 100, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 1D potential as arrays.
        
        Parameters:
        - n_points: Number of grid points
        - **kwargs: Additional parameters for the potential
        
        Returns:
        - (x_array, potential_array)
        """
        x = np.linspace(0, 1, n_points)
        V = self.evaluate_1d(x, **kwargs)
        return x, V
    
    def get_2d_array(self, n_points: int = 100, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 2D potential as arrays.
        
        Parameters:
        - n_points: Number of grid points in each direction
        - **kwargs: Additional parameters for the potential
        
        Returns:
        - (X, Y, V) where X, Y are meshgrids and V is the potential
        """
        x = np.linspace(0, 1, n_points)
        y = np.linspace(0, 1, n_points)
        X, Y = np.meshgrid(x, y)
        V = self.evaluate_2d(X, Y, **kwargs)
        return X, Y, V
    
    def plot_1d(self, n_points: int = 100, **kwargs):
        """Plot 1D potential."""
        x, V = self.get_1d_array(n_points, **kwargs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(x, V, 'b-', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('V(x)')
        plt.title(f'{self.name} Potential (1D)')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.show()
    
    def plot_2d(self, n_points: int = 100, **kwargs):
        """Plot 2D potential."""
        X, Y, V = self.get_2d_array(n_points, **kwargs)
        
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


class QuadraticPotential(Potential):
    """
    Quadratic potential: V(x) = k/2 * (x - x0)^2
    In 2D: V(x,y) = kx/2 * (x - x0)^2 + ky/2 * (y - y0)^2
    """
    
    def __init__(self):
        super().__init__("Quadratic")
    
    def _potential_function(self, *coords, k=1.0, center=None, kx=None, ky=None):
        """
        Parameters:
        - k: Spring constant (1D case)
        - center: Center position (1D: float, 2D: tuple)
        - kx, ky: Spring constants for x,y directions (2D case)
        """
        if len(coords) == 1:  # 1D case
            x = coords[0]
            x0 = center if center is not None else 0.5
            return 0.5 * k * (x - x0)**2
        
        elif len(coords) == 2:  # 2D case
            x, y = coords
            if center is None:
                x0, y0 = 0.5, 0.5
            else:
                x0, y0 = center
            
            kx_val = kx if kx is not None else k
            ky_val = ky if ky is not None else k
            
            return 0.5 * kx_val * (x - x0)**2 + 0.5 * ky_val * (y - y0)**2


class DoubleWellPotential(Potential):
    """
    Double well potential: V(x) = a * (x - x0)^4 - b * (x - x0)^2
    In 2D: V(x,y) = ax * (x - x0)^4 - bx * (x - x0)^2 + ay * (y - y0)^4 - by * (y - y0)^2
    """
    
    def __init__(self):
        super().__init__("Double Well")
    
    def _potential_function(self, *coords, a=1.0, b=1.0, center=None, ax=None, bx=None, ay=None, by=None):
        """
        Parameters:
        - a, b: Potential parameters (1D case)
        - center: Center position (1D: float, 2D: tuple)
        - ax, bx, ay, by: Potential parameters for x,y directions (2D case)
        """
        if len(coords) == 1:  # 1D case
            x = coords[0]
            x0 = center if center is not None else 0.5
            # Shift and scale to make it work well in [0,1]
            xi = (x - x0) * 2  # Scale to make wells visible
            return a * xi**4 - b * xi**2
        
        elif len(coords) == 2:  # 2D case
            x, y = coords
            if center is None:
                x0, y0 = 0.5, 0.5
            else:
                x0, y0 = center
            
            # Use provided parameters or defaults
            ax_val = ax if ax is not None else a
            bx_val = bx if bx is not None else b
            ay_val = ay if ay is not None else a
            by_val = by if by is not None else b
            
            # Scale coordinates
            xi = (x - x0) * 2
            yi = (y - y0) * 2
            
            return (ax_val * xi**4 - bx_val * xi**2) + (ay_val * yi**4 - by_val * yi**2)


class HarmonicOscillator2D(Potential):
    """
    2D Harmonic oscillator with possible coupling: V(x,y) = 1/2 * (kx*x^2 + ky*y^2 + kxy*x*y)
    """
    
    def __init__(self):
        super().__init__("2D Harmonic Oscillator")
    
    def _potential_function(self, *coords, kx=1.0, ky=1.0, kxy=0.0, center=None):
        """
        Parameters:
        - kx, ky: Spring constants in x,y directions
        - kxy: Coupling constant
        - center: Center position (tuple)
        """
        if len(coords) == 1:
            # 1D version - just harmonic oscillator
            x = coords[0]
            x0 = center if center is not None else 0.5
            return 0.5 * kx * (x - x0)**2
        
        elif len(coords) == 2:  # 2D case
            x, y = coords
            if center is None:
                x0, y0 = 0.5, 0.5
            else:
                x0, y0 = center
            
            dx = x - x0
            dy = y - y0
            
            return 0.5 * (kx * dx**2 + ky * dy**2 + kxy * dx * dy)


class CombinedPotential2D(Potential):
    """
    2D potential formed by combining two different 1D potentials:
    V(x,y) = V_x(x) + V_y(y)
    where V_x and V_y can be different potential types
    """
    
    def __init__(self, potential_x: Potential, potential_y: Potential):
        """
        Parameters:
        - potential_x: 1D potential for x-direction
        - potential_y: 1D potential for y-direction
        """
        self.potential_x = potential_x
        self.potential_y = potential_y
        super().__init__(f"Combined ({potential_x.name} + {potential_y.name})")
    
    def _potential_function(self, *coords, **kwargs):
        """
        Combine the two 1D potentials.
        kwargs are passed to both potentials - use prefixes 'x_' and 'y_' to specify direction-specific parameters
        """
        if len(coords) == 1:
            raise ValueError("CombinedPotential2D only supports 2D evaluation")
        
        elif len(coords) == 2:
            x, y = coords
            
            # Separate kwargs for x and y potentials
            x_kwargs = {}
            y_kwargs = {}
            common_kwargs = {}
            
            for key, value in kwargs.items():
                if key.startswith('x_'):
                    x_kwargs[key[2:]] = value  # Remove 'x_' prefix
                elif key.startswith('y_'):
                    y_kwargs[key[2:]] = value  # Remove 'y_' prefix
                else:
                    common_kwargs[key] = value
            
            # Merge common kwargs with specific ones
            x_kwargs.update(common_kwargs)
            y_kwargs.update(common_kwargs)
            
            V_x = self.potential_x.evaluate_1d(x, **x_kwargs)
            V_y = self.potential_y.evaluate_1d(y, **y_kwargs)
            
            return V_x + V_y
    
    def evaluate_1d(self, x: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
        """
        1D evaluation is not supported for combined 2D potentials.
        """
        raise ValueError("CombinedPotential2D only supports 2D evaluation")


# Convenience functions for easy access
def create_quadratic_potential():
    """Create a quadratic potential instance."""
    return QuadraticPotential()

def create_double_well_potential():
    """Create a double well potential instance."""
    return DoubleWellPotential()

def create_harmonic_oscillator_2d():
    """Create a 2D harmonic oscillator potential instance."""
    return HarmonicOscillator2D()

def create_combined_potential_2d(potential_x: Potential, potential_y: Potential):
    """Create a 2D potential by combining two 1D potentials."""
    return CombinedPotential2D(potential_x, potential_y)


"""
EXAMPLE: How to combine different 1D potentials into a 2D potential

# Create individual 1D potentials
quad = create_quadratic_potential()
dwell = create_double_well_potential()

# Combine them into a 2D potential: V(x,y) = V_quad(x) + V_dwell(y)
combined = create_combined_potential_2d(quad, dwell)

# Evaluate the combined potential
V = combined.evaluate_2d(0.3, 0.7, 
                        x_k=2.0, x_center=0.5,        # Parameters for quadratic in x
                        y_a=1.0, y_b=2.0, y_center=0.5)  # Parameters for double well in y

# Plot the combined potential
combined.plot_2d(x_k=2.0, x_center=0.5, y_a=1.0, y_b=2.0, y_center=0.5)

Note: Use prefixes 'x_' and 'y_' to specify parameters for each direction.
"""

# Example usage and testing
if __name__ == "__main__":
    print("Testing Potential Classes")
    print("=" * 40)
    
    # Test Quadratic Potential
    print("1. Quadratic Potential")
    quad = create_quadratic_potential()

    # Plot 1D and 2D potentials
    quad.plot_1d()
    quad.plot_2d()
    
    # Test Double Well Potential
    print("\n2. Double Well Potential")
    dwell = create_double_well_potential()

    # Plot 1D and 2D potentials
    dwell.plot_1d()
    dwell.plot_2d()

    # Test Combined Potential 2D (Quadratic x + Double Well y)
    print("\n4. Combined Potential 2D (Quadratic x + Double Well y)")
    combined = create_combined_potential_2d(quad, dwell)

    # Plot combined potential
    print("\n5. Plotting Combined Potential")
    combined.plot_2d(x_k=10.0, x_center=0.5, y_a=3.0, y_b=2.0, y_center=0.5)
    
    print("\nAll tests completed successfully!")

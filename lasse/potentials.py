import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Callable
from abc import ABC, abstractmethod


class Potential(ABC):
    """
    Abstract base class for potential functions.
    
    All potentials are defined on the domain [0,1] for both 1D and 2D.
    The potentials are designed to be zero at the boundaries:
    - 1D: V(0) = V(1) = 0
    - 2D: V(0,0) = V(0,1) = V(1,0) = V(1,1) = 0
    
    This ensures appropriate boundary conditions for quantum mechanical problems.
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
    Quadratic potential: V(x) = a * ((x - 0.5)^2 - 0.25), such that the potential is zero at 0 and 1, and depth = a / 4 (positive)
    In 2D: V(x,y) = a * ((x - 0.5)^2 + (y - 0.5)^2 - 0.5), such that the potential is zero at (0,0), (0,1), (1,0), (1,1), and depth = a / 2 (positive)
    """
    
    def __init__(self):
        super().__init__("Quadratic")
    
    def _potential_function(self, *coords, depth):
        """
        Parameters:
        - depth: Depth of the potential (positive value)
        
        Returns:
        - 1D: V(x) = 4*depth * ((x - 0.5)^2 - 0.25), zero at boundaries, minimum depth at center
        - 2D: V(x,y) = 2*depth * ((x - 0.5)^2 + (y - 0.5)^2 - 0.5), zero at corners, minimum depth at center
        """
        if len(coords) == 1:  # 1D case
            x = coords[0]
            a = 4 * depth
            return a * ((x - 0.5)**2 - 0.25)
        
        elif len(coords) == 2:  # 2D case
            x, y = coords
            a = 2 * depth
            
            return a * ((x - 0.5)**2 + (y - 0.5)**2 - 0.5)


class DoubleWellPotential(Potential):
    """
    Double well potential: V(x) = a * ((x - 0.5)^4 - 0.25 * (x - 0.5)^2), such that the potential is zero at 0, 0.5 and 1, and depth = a / 64 (positive)
    In 2D: V(x,y) = a * ((x - 0.5)^4 - 0.25 * (x - 0.5)^2 + (y - 0.5)^4 - 0.25 * (y - 0.5)^2), such that the potential is zero at (0,0), (0,1), (1,0), (1,1), and depth = a / 32 (positive)
    """
    
    def __init__(self):
        super().__init__("Double Well")
    
    def _potential_function(self, *coords, depth=1.0):
        """
        Parameters:
        - depth: Depth of the potential wells (positive value, default=1.0)
        
        Returns:
        - 1D: V(x) = 64*depth * ((x - 0.5)^4 - 0.25 * (x - 0.5)^2), zero at x=0, 0.5, 1 with wells at x≈0.146, 0.854
        - 2D: V(x,y) = 32*depth * (sum of 1D terms), zero at corners and center, with 4 wells
        """
        if len(coords) == 1:  # 1D case
            x = coords[0]
            a = depth * 64
            return a * ((x - 0.5)**4 - 0.25 * (x - 0.5)**2)
        
        elif len(coords) == 2:  # 2D case
            x, y = coords
            a = depth * 32

            return a * ((x - 0.5)**4 - 0.25 * (x - 0.5)**2 + (y - 0.5)**4 - 0.25 * (y - 0.5)**2)


class CombinedPotential2D(Potential):
    """
    2D potential formed by combining two different 1D potentials:
    V(x,y) = V_x(x) + V_y(y)
    
    This allows creating complex 2D potentials by combining simpler 1D components.
    For example, combine a quadratic potential in x with a double well potential in y.
    
    Parameters for each direction are specified using prefixes:
    - 'x_' prefix for x-direction potential parameters
    - 'y_' prefix for y-direction potential parameters
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
        Combine the two 1D potentials: V(x,y) = V_x(x) + V_y(y)
        
        Parameters:
        - **kwargs: Parameters for the individual potentials
          - Use 'x_' prefix for x-direction potential parameters (e.g., x_depth)
          - Use 'y_' prefix for y-direction potential parameters (e.g., y_depth)
          - Common parameters without prefix are passed to both potentials
        
        Example:
        - combined.evaluate_2d(x, y, x_depth=4.0, y_depth=10.0)
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


# Example usage and testing
if __name__ == "__main__":
    print("Testing Potential Classes")
    print("=" * 40)
    
    # Test Quadratic Potential
    print("1. Quadratic Potential")
    quad = QuadraticPotential()

    # Plot 1D and 2D potentials
    quad.plot_1d(depth=10.0)
    quad.plot_2d(depth=10.0)
    
    # Test Double Well Potential
    print("\n2. Double Well Potential")
    dwell = DoubleWellPotential()

    # Plot 1D and 2D potentials
    dwell.plot_1d(depth=10.0)
    dwell.plot_2d(depth=10.0)

    # Test Combined Potential 2D (Quadratic x + Double Well y)
    print("\n3. Combined Potential 2D (Quadratic x + Double Well y)")
    combined = CombinedPotential2D(quad, dwell)
    
    # Example evaluation
    V_test = combined.evaluate_2d(0.3, 0.7, x_depth=4.0, y_depth=10.0)
    print(f"Combined potential at (0.3, 0.7): {V_test}")

    # Plot combined potential
    print("\n4. Plotting Combined Potential")
    combined.plot_2d(x_depth=5.0, y_depth=5.0)
    
    print("\nAll tests completed successfully!")

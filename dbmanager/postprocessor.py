from __future__ import annotations
from typing import Callable
from ctypes import ArgumentError

from scipy.interpolate import LinearNDInterpolator
import numpy as np


from .simulation import SimulationReader


class Postprocessor:

    def __init__(self, sim: SimulationReader = None) -> None:
        self.sim = sim

    def register_sim(self, sim: SimulationReader = None) -> Postprocessor:
        self.sim = sim
        return self

    def linear_interpolator(self, data: np.array) -> Callable:
        return LinearNDInterpolator(self.sim.mesh[0], data)

    def integrate_axis(self, field, xi: np.array,
                       yi: np.array, axis: str = 'y') -> np.array:
        """Integrate in one axis for an array of positions.

        Args:
            field: str or function to evaluate (must be able to process array)
            xi: evaluation points x
            yi: evaluation points y
            axis: axis to integrate ('y', 'x')
        """
        if isinstance(field, str):
            field = self.linear_interpolator(self.sim.data(field, -1))
        if axis=='y':
            data = [np.trapz(field(xi, y), xi, axis=0) for y in yi]
        elif axis=='x':
            data = [np.trapz(field(x, yi), yi, axis=0) for x in xi]
        else:
            raise ArgumentError("Axis argument must be 'x' or 'y'")
        return np.array(data)

    def compute_principal_stress(self, arg):
        """Compute the principal stresses from the stress field.

        Args:
            arg: field name OR array (N, 4) or (N, 3)
        """
        if isinstance(arg, str):
            arr = self.sim._das(arg, -1)
        else:
            arr = arg

        if arr.shape[1]==4:
            midpoint = .5*(arr[:, 0]+arr[:, 3])
            radius = np.sqrt(((arr[:,0]-arr[:,3])/2)**2 + arr[:,1]**2)
        elif arr.shape[1]==3:
            midpoint = .5*(arr[:, 0]+arr[:, 2])
            radius = np.sqrt(((arr[:,0]-arr[:,2])/2)**2 + arr[:,1]**2)
        else:
            raise NotImplementedError("Don't know what to do...")

        return np.vstack((midpoint+radius, midpoint-radius, radius)).T




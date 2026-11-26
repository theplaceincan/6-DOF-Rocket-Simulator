"""Six-degree-of-freedom flight model of a small solid-motor rocket launched from a rail."""
from .config import Config
from .dynamics import FlightModel
from .simulate import simulate

__all__ = ["Config", "FlightModel", "simulate"]

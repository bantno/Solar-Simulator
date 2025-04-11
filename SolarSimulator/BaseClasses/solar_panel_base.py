# File: BaseClasses/solar_panel_base.py

import logging
from abc import ABC, ABCMeta, abstractmethod
from typing import Type, Dict

# Set up a logger for the module.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust the logging level as needed.


###############################################################################
# Metaclass for Automatic Registration
###############################################################################
class SolarPanelMeta(ABCMeta):
    """
    Metaclass for automatic registration of concrete SolarPanel subclasses.
    Any non-abstract subclass with a defined `model_name` attribute will
    automatically register itself with the SolarPanelFactory.
    """
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        # Only register classes that define a non-None model_name.
        # This avoids registering the abstract SolarPanel base class.
        model_name = getattr(cls, "model_name", None)
        if model_name is not None:
            SolarPanelFactory.register_panel(model_name, cls)
            logger.debug(f"Automatically registered solar panel model '{model_name}' for class {cls.__name__}.")


###############################################################################
# Solar Panel Base Interface
###############################################################################
class SolarPanel(ABC, metaclass=SolarPanelMeta):
    """
    Abstract base class for different solar panel models.
    
    Concrete implementations must define their own `area` and `efficiency` 
    properties and set the class variable `model_name` to a unique identifier.
    """
    model_name: str = None  # Unique identifier for registration; must be overridden.

    @property
    @abstractmethod
    def area(self) -> float:
        """
        Returns:
            float: The surface area of the solar panel in square meters.
        """
        pass

    @property
    @abstractmethod
    def efficiency(self) -> float:
        """
        Returns:
            float: The efficiency of the solar panel (a value between 0.0 and 1.0).
        """
        pass


###############################################################################
# Solar Panel Factory
###############################################################################
class SolarPanelFactory:
    """
    Factory class to instantiate solar panel models.
    
    Solar panel models are automatically registered via the SolarPanelMeta metaclass.
    """
    _registry: Dict[str, Type[SolarPanel]] = {}

    @classmethod
    def register_panel(cls, model_name: str, panel_class: Type[SolarPanel]) -> None:
        """
        Registers a new solar panel model by its unique name.

        Args:
            model_name (str): The unique identifier for the solar panel model.
            panel_class (Type[SolarPanel]): The class implementing the model.
        """
        if model_name in cls._registry:
            logger.warning(f"Solar panel model '{model_name}' is already registered. Overwriting registration.")
        cls._registry[model_name] = panel_class
        logger.debug(f"Registered solar panel model '{model_name}': {panel_class}")

    @classmethod
    def create_solar_panel(cls, model_name: str, **kwargs) -> SolarPanel:
        """
        Instantiates and returns a solar panel model based on the model name.

        Args:
            model_name (str): The registered name of the solar panel model.
            **kwargs: Additional keyword arguments passed to the model's constructor.

        Returns:
            SolarPanel: An instance of the requested solar panel model.

        Raises:
            ValueError: If the model_name is not found in the registry.
        """
        if model_name not in cls._registry:
            raise ValueError(f"Unknown solar panel model '{model_name}'")
        panel_class = cls._registry[model_name]
        logger.debug(f"Creating solar panel model '{model_name}' with parameters {kwargs}")
        return panel_class(**kwargs)


###############################################################################
# Concrete Solar Panel Models
###############################################################################
class ConstantSolarPanel(SolarPanel):
    """
    A simple solar panel model with a fixed area and efficiency.
    """
    model_name: str = "constant"

    def __init__(self, area: float = 0.65, efficiency: float = 0.1):
        self._area = area
        self._efficiency = efficiency

    @property
    def area(self) -> float:
        return self._area

    @property
    def efficiency(self) -> float:
        return self._efficiency

class VariableEfficiencySolarPanel(SolarPanel):
    """
    A solar panel model with efficiency that degrades over time.
    
    The efficiency is computed based on a base efficiency and a degradation 
    factor applied incrementally over successive calls.
    """
    model_name: str = "variable"

    def __init__(self, area: float = 0.65, base_efficiency: float = 0.1, degradation: float = 0.001):
        self._area = area
        self._base_efficiency = base_efficiency
        self._degradation = degradation
        self._time = 0  # Internal counter to simulate degradation over time

    @property
    def area(self) -> float:
        return self._area

    @property
    def efficiency(self) -> float:
        # Compute the current efficiency based on elapsed time.
        current_efficiency = max(0.0, self._base_efficiency - self._degradation * self._time)
        self._time += 1  # Increment time for the next call.
        return current_efficiency



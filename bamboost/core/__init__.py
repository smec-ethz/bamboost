import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach(
    __name__,
    [],
    {
        "collection": ["Collection"],
        "simulation": ["Simulation", "SimulationWriter", "FieldType"],
    },
)

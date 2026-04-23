__all__ = ["Lite3System"]


def __getattr__(name: str):
    if name == "Lite3System":
        from lite3_system.system import Lite3System

        return Lite3System
    raise AttributeError(name)

__all__ = ["CPU"]


def __getattr__(name: str):
    if name == "CPU":
        from .cpu import CPU

        return CPU
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

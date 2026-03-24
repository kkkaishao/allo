from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..core.kernel import Kernel


class Backend(ABC):

    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    @abstractmethod
    def prerequisite_check(self):
        """Check if the necessary tools and environment are set up for this backend."""
        raise NotImplementedError()

    @abstractmethod
    def scaffolding_project(self, prj_dir: str):
        """Generate a scaffolding project for the kernel."""
        raise NotImplementedError()

    @abstractmethod
    def run_synth(self):
        raise NotImplementedError()

    @abstractmethod
    def run_csim(self):
        raise NotImplementedError()

    @abstractmethod
    def run_cosim(self):
        raise NotImplementedError()

    @abstractmethod
    def cleanup(self):
        raise NotImplementedError()

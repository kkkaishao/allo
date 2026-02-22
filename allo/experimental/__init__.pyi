from .dsl import *
from .core import *

from . import settings

class PartitionKind(Enum):
    Complete = 0
    Cyclic = 1
    Block = 2

Complete = PartitionKind.Complete
Cyclic = PartitionKind.Cyclic
Block = PartitionKind.Block

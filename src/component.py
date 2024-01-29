from dataclasses import dataclass, field
from typing import List, Tuple
from pigframe.world import Component

@dataclass
class Mouse(Component):
    x: float
    y: float

@dataclass
class Graph(Component):
    pos_list: List[Tuple] = field(default_factory=list)
    
@dataclass
class PrevGraph(Graph):
    y_list: List[Tuple] = field(default_factory=list)
    
@dataclass
class Neuron(Component):
    w: float
    b: float
    editing: bool = False
    
@dataclass
class Layer(Component):
    neuron_ids: List[int] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
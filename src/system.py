from pigframe.world import System
import pyxel
import numpy as np
from sympy import *
import math

from component import *


x = Symbol('x')
z = Symbol('z')
y = Symbol('y')

def sigmoid(x_):
    return 1/(1+math.exp(-x_))

class SysUpdatePrevGraph(System):
    def __init__(self, world, priority: int = 0, *args) -> None:
        super().__init__(world, priority, *args)
        
    def process(self):
        x_min = self.world.x_min
        y_min = self.world.y_min
        x_max = self.world.x_max
        y_max = self.world.y_max
        x_scale = self.world.x_scale
        y_scale  = self.world.y_scale
        screen_size = self.world.SCREEN_SIZE
        X = np.linspace(x_min, x_max, 10000)
        z_all = [0.0 for x_ in X]
        y_all = [0.0 for x_ in X]
        
        for ent, (prev_graph) in self.world.get_component(PrevGraph):
            prev_graph.y_list = []
            
            for ent, (layer) in self.world.get_component(Layer):
                weights = layer.weights

                for i, neuron_id in enumerate(layer.neuron_ids):
                    neuron = self.world.get_entity_object(neuron_id)[Neuron]
                    Z = [neuron.w * x_ + neuron.b for x_ in X]
                    Y = [sigmoid(z_) * weights[i] for z_ in Z]
                    y_all = np.add(y_all, Y)
                    
            prev_graph.y_list = y_all.tolist()
            
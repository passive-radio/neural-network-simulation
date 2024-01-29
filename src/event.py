from pigframe.world import Event
import pyxel
import numpy as np
from sympy import *
import math

from datetime import datetime, timedelta, timezone

from component import *

x = Symbol('x')
z = Symbol('z')
y = Symbol('y')

def sigmoid(x_):
    return 1/(1+math.exp(-x_))

class EvPenDown(Event):
    def _Event__process(self):
        mouse_pos = (pyxel.mouse_x, pyxel.mouse_y)
        for ent, (graph) in self.world.get_component(Graph):
            graph.pos_list.append(mouse_pos)
            
    
class EvEditNeuron(Event):
    def __init__(self, world, priority: int = 0, *args) -> None:
        super().__init__(world, priority, *args)
        self.collision_detection_threshold = 25
        
    def _Event__process(self):
        x_min = self.world.x_min
        y_min = self.world.y_min
        x_max = self.world.x_max
        y_max = self.world.y_max
        x_scale = self.world.x_scale
        y_scale  = self.world.y_scale
        screen_size = self.world.SCREEN_SIZE
        X = np.linspace(x_min, x_max, 10000)
        
        mouse_pos = (pyxel.mouse_x, pyxel.mouse_y)
        for ent, (layer) in self.world.get_component(Layer):
            weights = layer.weights
            neuron_ids = layer.neuron_ids
            
            for i, neuron_id in enumerate(neuron_ids):
                neuron = self.world.get_entity_object(neuron_id)[Neuron]
                
                # 自分もほかのニューロンも編集していない場合のみ編集を許可する
                print(neuron.editing, self.world.editing_neuron)
                if (neuron.editing == False) and (self.world.editing_neuron is None):
                    Z  = [neuron.w * x_ + neuron.b for x_ in X]
                    Y = [(y_max - sigmoid(z_) * weights[i])* y_scale + self.world.top for z_ in Z]
                    
                    X_diff_abs = np.abs(X * x_scale + self.world.left - mouse_pos[0])
                    X_diff_abs_argmin = np.argmin(X_diff_abs)
                    X_diff_abs_min = np.min(X_diff_abs)
                    
                    if X_diff_abs_min < self.collision_detection_threshold:
                        Y_diff = Y[X_diff_abs_argmin] - mouse_pos[1]
                        if -1 * self.collision_detection_threshold < Y_diff < self.collision_detection_threshold:
                            neuron.editing = True
                            self.world.editing_neuron = neuron_id
                            print(f"Neuron Editing: {i}")
                
                elif neuron.editing == True:
                    # s = -b / w
                    new_s = (mouse_pos[0] - self.world.left) / x_scale
                    new_b = - new_s * neuron.w
                    neuron.b = new_b
                    
                    weights[i] = (y_max * y_scale + self.world.top - mouse_pos[1]) / y_scale
                    
class EvUpdateNeuron(Event):
    def _Event__process(self):
        for ent, (layer) in self.world.get_component(Layer):
            weights = layer.weights
            neuron_ids = layer.neuron_ids
            
            for i, neuron_id in enumerate(neuron_ids):
                neuron = self.world.get_entity_object(neuron_id)[Neuron]
                self.world.editing_neuron = None
                if neuron.editing:
                    neuron.editing = False
                    print(f"Neuron Edited: {i}")
                    print(f"Editing Neuron: {self.world.editing_neuron}")
                    
class EvAddNeuron(Event):
    def _Event__process(self):
        ent_neuron = self.world.create_entity()
        self.world.add_component_to_entity(ent_neuron, Neuron, 60, -40)

        for ent, (layer) in self.world.get_component(Layer):
            layer.neuron_ids.append(ent_neuron)
            layer.weights.append(0.2)

        print(f"Neuron Added: {len(layer.neuron_ids) - 1}")
        
class EvReportNetwork(Event):
    def __init__(self, world, priority: int = 0, **kwargs) -> None:
        super().__init__(world, priority)
        if kwargs.get("filepath") is None:
            raise ValueError("filepath is not set")
        self.filepath = kwargs["filepath"]
        
    def _Event__process(self):
        out = ""
        now = datetime.now(timezone(timedelta(hours=9)))
        out += f"Reported at {now.strftime('%Y-%m-%d %H:%M:%S')}"
        for ent, (layer) in self.world.get_component(Layer):
            weights = layer.weights
            neuron_ids = layer.neuron_ids
            count_neurons = len(neuron_ids)
            neurons = []
            out += f"\nlayer {layer.id} ------------------\n"
            for i, neuron_id in enumerate(neuron_ids):
                neuron = self.world.get_entity_object(neuron_id)[Neuron]
                neurons.append(neuron)
                out += f"neuron {i}: w = {int(neuron.w)}, b = {int(neuron.b)}, sigmoid_w = {weights[i]}\n"
        
            out += "Tex -------------------\n"
            out += f"$$\n\\begin{{align}}\n"
            out += f"L_{i} & = \sum_{{i=1}}^{count_neurons} v_i \cdot \sigma(w_i x + b_i) \\\\ \n"
            out += "& = "
            summation_parts = [f"({weights[i]} \cdot \sigma({int(neuron.w)} x + {int(neuron.b)})) \\\\\n&" for i, neuron in enumerate(neurons)]
            out += f"{' + '.join(summation_parts)}"
            out = out[:-1]
            out += f"\\end{{align}}\n$$"
        filename = self.filepath.split(".")[-2]
        ext = self.filepath.split(".")[1]
        filename += now.strftime('%Y%m%d%H%M%S')
        filepath = f"{filename}.{ext}"
        
        with open(filepath, mode="w") as f:
            f.write(out)
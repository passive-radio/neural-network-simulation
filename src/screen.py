from pigframe.world import Screen
import pyxel
import math
import numpy as np
from sympy import *

from component import *
from font import BDFRenderer

x = Symbol('x')
z = Symbol('z')
y = Symbol('y')

def button_pressed(x: int, y: int, w: int, h: int) -> bool:
    return (x < pyxel.mouse_x < x + w) and (y < pyxel.mouse_y < y + h) and pyxel.btnp(pyxel.MOUSE_BUTTON_LEFT)

def sigmoid(x_):
    return 1/(1+math.exp(-x_))

class ScInitialGuide(Screen):
    def __init__(self, world, priority: int = 0, *args) -> None:
        super().__init__(world, priority, *args)
        
    def draw(self):
        self.world.font.draw_text(200, 10, "Click and drag mouse to make initial graph", 1)
        self.world.font.draw_text(200, 30, "Press Enter to start simulation!", 1)

class ScDrawGraph(Screen):
    def __init__(self, world, priority: int = 0, *args) -> None:
        super().__init__(world, priority, *args)
        self.padding_left = 0
        self.padding_bottom = 0

    def draw(self):
        for ent, (graph) in self.world.get_component(Graph):
            for pos in graph.pos_list:
                pyxel.rect(pos[0], pos[1], 2, 2, 1)
                pyxel.rect(pyxel.mouse_x, pyxel.mouse_y, 2, 2, 4)
                
        self.world.font.draw_text(self.world.x_min * self.world.x_scale + self.world.left, (self.world.y_max - self.world.y_min) * self.world.y_scale + self.world.top, "0", 1)
        self.world.font.draw_text(self.world.x_max * self.world.x_scale + self.world.left, (self.world.y_max - self.world.y_min) * self.world.y_scale + self.world.top, str(self.world.x_max), 1)
        self.world.font.draw_text(self.world.x_min * self.world.x_scale + self.world.left, self.world.top, str(self.world.y_max), 1)

        pyxel.line(self.world.x_min * self.world.x_scale + self.world.left + self.padding_left, (self.world.y_max - self.world.y_min) * self.world.y_scale + self.world.top - self.padding_bottom, self.world.x_max * self.world.x_scale + self.world.left - self.padding_bottom, (self.world.y_max - self.world.y_min) * self.world.y_scale + self.world.top - self.padding_bottom, 1)
        pyxel.line(self.world.x_min * self.world.x_scale + self.world.left + self.padding_left, self.world.top + 2, self.world.x_min * self.world.x_scale + self.world.left + self.padding_left, (self.world.y_max - self.world.y_min) * self.world.y_scale + self.world.top - self.padding_bottom, 1)
                
class ScGraph(Screen):
    def __init__(self, world, priority: int = 0, *args) -> None:
        super().__init__(world, priority, *args)
        self.padding_left = 0
        self.padding_bottom = 0
        
    def draw(self):
        for ent, (graph) in self.world.get_component(Graph):
            for pos in graph.pos_list:
                pyxel.rect(pos[0], pos[1], 1, 1, 1)
                
        self.world.font.draw_text(self.world.x_min * self.world.x_scale + self.world.left, (self.world.y_max - self.world.y_min) * self.world.y_scale + self.world.top, "0", 1)
        self.world.font.draw_text(self.world.x_max * self.world.x_scale + self.world.left, (self.world.y_max - self.world.y_min) * self.world.y_scale + self.world.top, str(self.world.x_max), 1)
        self.world.font.draw_text(self.world.x_min * self.world.x_scale + self.world.left, self.world.top, str(self.world.y_max), 1)
        
        pyxel.line(self.world.x_min * self.world.x_scale + self.world.left + self.padding_left, (self.world.y_max - self.world.y_min) * self.world.y_scale + self.world.top - self.padding_bottom, self.world.x_max * self.world.x_scale + self.world.left - self.padding_bottom, (self.world.y_max - self.world.y_min) * self.world.y_scale + self.world.top - self.padding_bottom, 1)
        pyxel.line(self.world.x_min * self.world.x_scale + self.world.left + self.padding_left, self.world.top + 2, self.world.x_min * self.world.x_scale + self.world.left + self.padding_left, (self.world.y_max - self.world.y_min) * self.world.y_scale + self.world.top - self.padding_bottom, 1)

class ScSimulationParams(Screen):
    def __init__(self, world, priority: int = 0, *args) -> None:
        super().__init__(world, priority, *args)
        self.left = 60
        self.top = 160
        
    def draw(self):
        for ent, (layer) in self.world.get_component(Layer):
            weights = layer.weights
            for i, neuron_id in enumerate(layer.neuron_ids):
                neuron = self.world.get_entity_object(neuron_id)[Neuron]
                self.world.font.draw_text(int(self.world.x_max + self.world.x_scale + self.left), int(self.world.y_min * self.world.y_scale + i * 40 + self.top),
                                    f"ニューロン {i}: w = {int(neuron.w)}, b = {int(neuron.b)}, シグモイド重みv = {weights[i]}", 0)
        
class ScApprox(Screen):
    """_summary_

    Parameters
    ----------
    Screen : _type_
        _description_
    """
    
    def draw(self):
        x_min = self.world.x_min
        y_min = self.world.y_min
        x_max = self.world.x_max
        y_max = self.world.y_max
        x_scale = self.world.x_scale
        y_scale  = self.world.y_scale
        screen_size = self.world.SCREEN_SIZE
        X = np.linspace(x_min, x_max, 10000)
        
        for  ent, (prev_graph) in self.world.get_component(PrevGraph):
            for i, y in enumerate(prev_graph.y_list):
                pyxel.rect(int(X[i] * x_scale + self.world.left), int((y_max - y) * y_scale + self.world.top), 3, 3, 1)
                
        for ent, (layer) in self.world.get_component(Layer):
            weights = layer.weights
            
            for i, neuron_id in enumerate(layer.neuron_ids):
                neuron = self.world.get_entity_object(neuron_id)[Neuron]
                Z = [neuron.w * x_ + neuron.b for x_ in X]
                Y = [(y_max - sigmoid(z_) * weights[i])* y_scale + self.world.top for z_ in Z]
                for i, y in enumerate(Y):
                    if i % 40 in (0, 1, 2, 3, 4, 5, 6, 1, 8):
                        pyxel.rect(int(X[i] * x_scale + self.world.left), int(y), 2, 2, 4)

class ScAddNeuronButton(Screen):
    def __init__(self, world, priority: int = 0, *args) -> None:
        super().__init__(world, priority, *args)
        self.world.triger_add_neuron = lambda: button_pressed(self.world.x_max * self.world.x_scale + self.world.left + 40, self.world.y_min * self.world.y_scale + 40, 120, 30)

    def draw(self):
        pyxel.rect(self.world.x_max * self.world.x_scale + self.world.left + 40, self.world.y_min * self.world.y_scale + 40, 120, 30, 11)
        self.world.font.draw_text(self.world.x_max * self.world.x_scale + self.world.left + 45, self.world.y_min * self.world.y_scale + 42, "Add Neuron!", 0)
        
class ScButtonReportNetwork(Screen):
    def __init__(self, world, priority: int = 0, *args) -> None:
        super().__init__(world, priority, *args)
        self.world.triget_report_network = lambda: button_pressed(self.world.x_max * self.world.x_scale + self.world.left + 200, self.world.y_min * self.world.y_scale + 40, 120, 30)
    
    def draw(self):
        pyxel.rect(self.world.x_max * self.world.x_scale + self.world.left + 200, self.world.y_min * self.world.y_scale + 40, 120, 30, 11)
        self.world.font.draw_text(self.world.x_max * self.world.x_scale + self.world.left + 205, self.world.y_min * self.world.y_scale + 42, "Report Network!", 0)
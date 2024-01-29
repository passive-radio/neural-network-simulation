from pigframe.world import *
import pyxel

from component import *
from event import *
from system import *
from screen import *

class App(World):
    def __init__(self, x_min=0, y_min=0, x_max = 1, y_max = 1, x_scale = 900, y_scale = 400, screen_size = (1440, 1000), top: int = 40, left: int = 20):
        super().__init__()
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.SCREEN_SIZE = screen_size
        self.top = top
        self.left = left
        self.FPS = 90
        pyxel.init(self.SCREEN_SIZE[0], self.SCREEN_SIZE[1], title="Neural Netowrk Simulator", fps=self.FPS, capture_scale=2)
        self.font = BDFRenderer("assets/umplus_j12r.bdf")
        self.init()

    def init(self):
        pyxel.mouse(True)
        # editing: 編集中のニューロン entity_id。いずれのニューロンも編集中でない場合は None
        self.editing_neuron = None
    
    def run(self):
        pyxel.run(self.update, self.draw)
        
    def update(self):
        self.process()
        
    def draw(self):
        pyxel.cls(7)
        self.process_screens()
        
if __name__ == "__main__":
    app = App()

    app.add_scenes(["launch", "make-initial-graph", "simulation"])

    neuron1 = app.create_entity()
    neuron2 = app.create_entity()
    layer1 = app.create_entity()
    ent_graph_true = app.create_entity()
    ent_graph_prev = app.create_entity()
    
    app.add_screen_to_scenes(ScDrawGraph, "make-initial-graph", 1)
    app.add_screen_to_scenes(ScInitialGuide, "make-initial-graph", 0)
    app.add_screen_to_scenes(ScGraph, "simulation", 2)
    app.add_screen_to_scenes(ScApprox, "simulation", 1)
    app.add_screen_to_scenes(ScSimulationParams, "simulation", 0)
    app.add_screen_to_scenes(ScAddNeuronButton, "simulation", 3)
    app.add_screen_to_scenes(ScButtonReportNetwork, "simulation", 4)
    
    app.add_event_to_scene(EvPenDown, "make-initial-graph", triger=lambda: pyxel.btn(pyxel.MOUSE_BUTTON_LEFT), priority=0)
    app.add_event_to_scene(EvEditNeuron, "simulation", triger=lambda: pyxel.btn(pyxel.MOUSE_BUTTON_LEFT), priority=1)
    app.add_event_to_scene(EvUpdateNeuron, "simulation", triger=lambda: pyxel.btnr(pyxel.MOUSE_BUTTON_LEFT), priority=0)
    app.add_event_to_scene(EvAddNeuron, "simulation", triger=app.triger_add_neuron, priority=2)
    app.add_event_to_scene(EvReportNetwork, "simulation", triger=app.triget_report_network, priority=3, filepath = "network.txt")
    
    app.add_component_to_entity(ent_graph_true, Graph)
    app.add_component_to_entity(ent_graph_prev, PrevGraph)
    app.add_component_to_entity(layer1, Layer, 1, [neuron1, neuron2], [0.6, -0.6])
    app.add_component_to_entity(neuron1, Neuron, 100, -40) # s = -b / w
    app.add_component_to_entity(neuron2, Neuron, 50, -40) # s = -b / w
    
    app.add_system_to_scenes(SysUpdatePrevGraph, "simulation", 0)
    
    app.add_scene_transition("make-initial-graph", "simulation", lambda: pyxel.btnp(pyxel.KEY_RETURN))

    app.current_scene = "make-initial-graph"
    app.run()
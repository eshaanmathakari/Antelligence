from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from ant_model import ForagingModel, AntAgent, Food

def agent_portrayal(agent):
    if isinstance(agent, AntAgent):
        color = "blue" if agent.carrying_food else "red"
        return {"Shape": "circle", "r": 0.5, "Color": color, "Layer": 1}
    elif isinstance(agent, Food):
        return {"Shape": "rect", "w": 0.4, "h": 0.4, "Color": "green", "Layer": 0}

grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)
chart = ChartModule([{"Label": "Carrying", "Color": "Black"}])

server = ModularServer(ForagingModel,
                       [grid, chart],
                       "Ant Foraging with GPT",
                       {"width": 20, "height": 20, "N_ants": 10, "N_food": 20})

server.port = 8521
server.launch()

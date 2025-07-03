import matplotlib.pyplot as plt
import numpy as np
from random import choice, random
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for file output
from matplotlib.animation import FuncAnimation

# Define Ant Agent
class AntAgent:
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
        self.carrying_food = False
        self.pos = (np.random.randint(model.width), np.random.randint(model.height))

    def step(self):
        # Random movement (for now)
        x, y = self.pos
        possible_steps = self.model.get_neighborhood(x, y)
        new_position = choice(possible_steps)
        self.pos = new_position
        
        # Pickup food if available
        if self.model.is_food_at(self.pos) and not self.carrying_food:
            self.carrying_food = True
            self.model.remove_food(self.pos)

        # Drop food randomly after carrying for a while
        if self.carrying_food and random() < 0.05:
            self.carrying_food = False
            self.model.place_food(self.pos)

class ForagingModel:
    def __init__(self, width, height, N_ants, N_food):
        self.width = width
        self.height = height
        self.ants = [AntAgent(i, self) for i in range(N_ants)]
        self.foods = [(np.random.randint(width), np.random.randint(height)) for _ in range(N_food)]

    def step(self):
        for ant in self.ants:
            ant.step()

    def get_neighborhood(self, x, y):
        # Return 8 possible adjacent cells, excluding the current cell (Moore neighborhood)
        return [(i, j) for i in range(x-1, x+2) for j in range(y-1, y+2) if (i, j) != (x, y) and 0 <= i < self.width and 0 <= j < self.height]

    def is_food_at(self, pos):
        return pos in self.foods

    def remove_food(self, pos):
        if pos in self.foods:
            self.foods.remove(pos)

    def place_food(self, pos):
        if pos not in self.foods:
            self.foods.append(pos)

    def get_agent_positions(self):
        return [ant.pos for ant in self.ants]

    def get_food_positions(self):
        return self.foods

def plot_grid(model, step):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, model.width)
    ax.set_ylim(0, model.height)

    # Plot ants (blue if not carrying food, red if carrying food)
    ant_positions = model.get_agent_positions()
    for i, pos in enumerate(ant_positions):
        carrying_food = model.ants[i].carrying_food
        color = 'red' if carrying_food else 'blue'
        ax.plot(pos[0], pos[1], 'o', color=color, markersize=10)

    # Plot food
    food_positions = model.get_food_positions()
    for pos in food_positions:
        ax.plot(pos[0], pos[1], 'go', markersize=6)

    ax.set_title(f'Ant Foraging Simulation - Step {step}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.grid(True)
    
    # Save plot instead of showing
    plt.savefig(f'ant_simulation_step_{step}.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

# Run the model and plot the grid
if __name__ == "__main__":
    model = ForagingModel(width=20, height=20, N_ants=10, N_food=20)

    # Run 10 steps and visualize the grid
    for step in range(10):
        model.step()
        plot_grid(model, step)
        
    print("Simulation completed! Check the generated PNG files.")

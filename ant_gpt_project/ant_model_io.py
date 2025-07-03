import os
from dotenv import load_dotenv
import openai
import matplotlib.pyplot as plt
import numpy as np
from random import choice, random
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for file output

# Load API key from .env
load_dotenv()
IO_API_KEY = os.getenv("IO_SECRET_KEY")

def ask_io_for_ant_decision(ant, model, io_client):
    """
    Query the IO Intelligence API to decide the ant's next move.
    Returns: 'toward', 'random', or 'stay'
    """
    x, y = ant.pos
    # Check if food is in the Moore neighborhood
    food_nearby = any(
        abs(fx - x) <= 1 and abs(fy - y) <= 1
        for fx, fy in model.get_food_positions()
    )
    carrying = ant.carrying_food

    user_prompt = (
        f"You are an ant at position {ant.pos} on a {model.width}x{model.height} grid. "
        f"Food nearby: {food_nearby}. Carrying food: {carrying}. "
        "Should you move toward food, move randomly, or stay? "
        "Reply with 'toward', 'random', or 'stay'."
    )

    response = io_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",  # Or another available model
        messages=[
            {"role": "system", "content": "You are an ant foraging for food."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        stream=False,
        max_completion_tokens=10
    )
    action = response.choices[0].message.content.strip().lower()
    return action

class ForagingModel:
    def __init__(self, width, height, N_ants, N_food):
        self.width = width
        self.height = height
        # FIX: foods must be initialized before ants
        self.foods = [(np.random.randint(width), np.random.randint(height)) for _ in range(N_food)]
        self.io_client = openai.OpenAI(
            api_key=IO_API_KEY,
            base_url="https://api.intelligence.io.solutions/api/v1/"
        )
        self.ants = [AntAgent(i, self) for i in range(N_ants)]

    def step(self):
        for ant in self.ants:
            ant.step()

    def get_neighborhood(self, x, y):
        # Return 8 possible adjacent cells, excluding the current cell (Moore neighborhood)
        return [(i, j) for i in range(x-1, x+2) for j in range(y-1, y+2)
                if (i, j) != (x, y) and 0 <= i < self.width and 0 <= j < self.height]

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

class AntAgent:
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
        self.carrying_food = False
        self.pos = (np.random.randint(model.width), np.random.randint(model.height))

    def step(self):
        x, y = self.pos
        possible_steps = self.model.get_neighborhood(x, y)
        action = ask_io_for_ant_decision(self, self.model, self.model.io_client)

        if action == "toward":
            # Move to adjacent cell with food if possible
            food_cells = [pos for pos in possible_steps if self.model.is_food_at(pos)]
            if food_cells:
                new_position = food_cells[0]
            else:
                new_position = choice(possible_steps)
        elif action == "random":
            new_position = choice(possible_steps)
        elif action == "stay":
            new_position = self.pos
        else:
            new_position = choice(possible_steps)  # fallback

        self.pos = new_position

        # Pickup food if available
        if self.model.is_food_at(self.pos) and not self.carrying_food:
            self.carrying_food = True
            self.model.remove_food(self.pos)

        # Drop food randomly after carrying for a while
        if self.carrying_food and random() < 0.05:
            self.carrying_food = False
            self.model.place_food(self.pos)

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

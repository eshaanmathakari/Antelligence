import os
from dotenv import load_dotenv
import openai
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for file output
import matplotlib.pyplot as plt
import numpy as np
from random import choice, random
import json # for queen ant if used here

# Load API key from .env
load_dotenv()
IO_API_KEY = os.getenv("IO_SECRET_KEY")

def _step_toward(start, target, model):
    x, y = start
    tx, ty = target
    # Ensure all neighbors are valid (within bounds)
    valid_neighbors = model.get_neighborhood(x,y) + [start] # Include current position as an option
    return min(valid_neighbors, key=lambda n: abs(n[0]-tx)+abs(n[1]-ty))


def ask_io_for_ant_decision(ant, model, io_client, selected_model_name):
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

    try:
        response = io_client.chat.completions.create(
            model=selected_model_name,  # Use the passed model name
            messages=[
                {"role": "system", "content": "You are an ant foraging for food. Reply with one word: toward, random, or stay."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            stream=False,
            max_completion_tokens=10
        )
        action = response.choices[0].message.content.strip().lower()
        return action if action in ["toward", "random", "stay"] else "random"
    except Exception as e:
        print(f"API call failed for ant {ant.unique_id}: {str(e)}. Falling back to random.")
        return "random"

class ForagingModel:
    def __init__(self, width, height, N_ants, N_food, use_queen=False, use_llm_queen=False):
        self.width = width
        self.height = height
        self.foods = []
        while len(self.foods) < N_food:
            new_food_pos = (np.random.randint(width), np.random.randint(height))
            if new_food_pos not in self.foods:
                self.foods.append(new_food_pos)

        self.io_client = openai.OpenAI(
            api_key=IO_API_KEY,
            base_url="https://api.intelligence.io.solutions/api/v1/"
        ) if IO_API_KEY else None
        self.selected_model = "meta-llama/Llama-3.3-70B-Instruct" # Default model

        self.ants = [AntAgent(i, self) for i in range(N_ants)]
        self.queen = QueenAnt(self, use_llm=use_llm_queen) if use_queen else None

    def step(self):
        guidance = {}
        if self.queen:
            guidance = self.queen.guide()

        for ant in self.ants:
            # Pass the suggested cell (or None)
            ant.step(guidance.get(ant))

    def get_neighborhood(self, x, y):
        neigh = [(x+dx, y+dy)
                 for dx in (-1,0,1)
                 for dy in (-1,0,1)
                 if (dx,dy)!=(0,0)]
        valid_neigh = []
        for i,j in neigh:
            if 0 <= i < self.width and 0 <= j < self.height and (i,j) not in valid_neigh:
                valid_neigh.append((i,j))
        return valid_neigh

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

    def step(self, guided_pos=None):
        x, y = self.pos
        possible_steps = self.model.get_neighborhood(x, y)
        new_position = self.pos

        if guided_pos:
            new_position = guided_pos
        else:
            action = ask_io_for_ant_decision(self, self.model, self.model.io_client, self.model.selected_model)

            if action == "toward":
                # Move toward nearest food
                target_food = self._find_nearest_food()
                if target_food:
                    new_position = _step_toward(self.pos, target_food, self.model)
                else:
                    new_position = choice(possible_steps) if possible_steps else self.pos
            elif action == "random":
                new_position = choice(possible_steps) if possible_steps else self.pos
            elif action == "stay":
                new_position = self.pos
            else:
                new_position = choice(possible_steps) if possible_steps else self.pos # fallback

        self.pos = new_position

        # Pickup food if available
        if self.model.is_food_at(self.pos) and not self.carrying_food:
            self.carrying_food = True
            self.model.remove_food(self.pos)

        # Drop food randomly after carrying for a while
        if self.carrying_food and random() < 0.05:
            self.carrying_food = False
            self.model.place_food(self.pos)

    def _find_nearest_food(self):
        if not self.model.foods:
            return None
        return min(self.model.foods,
                   key=lambda f: abs(f[0]-self.pos[0]) + abs(f[1]-self.pos[1]))


# ---------------- Queen agent (for ant_model_io.py) ---------------- #
class QueenAnt:
    def __init__(self, model, use_llm=False):
        self.model = model
        self.use_llm = use_llm

    def guide(self) -> dict:
        guidance = {}
        if not self.model.foods:
            return guidance

        if self.use_llm and self.model.io_client:
            return self._guide_with_llm()
        else:
            return self._guide_with_heuristic()

    def _guide_with_heuristic(self) -> dict:
        guidance = {}
        ants = self.model.ants
        foods = self.model.foods
        for ant in ants:
            target = min(foods, key=lambda f: abs(f[0]-ant.pos[0]) + abs(f[1]-ant.pos[1]))
            best_step = _step_toward(ant.pos, target, self.model)
            guidance[ant] = best_step
        return guidance

    def _guide_with_llm(self) -> dict:
        guidance = {}
        ant_data = []
        for i, ant in enumerate(self.model.ants):
            ant_data.append({
                "id": ant.unique_id,
                "position": list(ant.pos),
                "carrying_food": ant.carrying_food
            })

        state = {
            "ants": ant_data,
            "food_positions": [list(p) for p in self.model.foods],
            "grid_size": [self.model.width, self.model.height]
        }

        system_prompt = (
            "You are a hyper-intelligent Queen Ant. Your goal is to optimize food collection for your colony. "
            "Given the current state of ants and food, you need to provide a single best next step for each ant. "
            "For each ant, select one adjacent cell (including diagonals) or its current cell. "
            "Output a JSON object where keys are ant IDs and values are their chosen new [x, y] coordinates. "
            "Example: {\"0\": [5,6], \"1\": [10,12]}"
        )
        user_prompt = f"Current state: {json.dumps(state)}"

        try:
            response = self.model.io_client.chat.completions.create(
                model=self.model.selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_completion_tokens=500
            )
            llm_response_content = response.choices[0].message.content.strip()
            moves = json.loads(llm_response_content)

            for ant_id_str, cell in moves.items():
                ant_id = int(ant_id_str)
                ant_obj = next((ant for ant in self.model.ants if ant.unique_id == ant_id), None)
                if ant_obj and isinstance(cell, list) and len(cell) == 2:
                    proposed_pos = tuple(cell)
                    valid_moves = self.model.get_neighborhood(*ant_obj.pos) + [ant_obj.pos]
                    if proposed_pos in valid_moves:
                        guidance[ant_obj] = proposed_pos
        except json.JSONDecodeError as e:
            print(f"Queen LLM response was not valid JSON: {e}. Response: {llm_response_content}. Falling back to heuristic guidance.")
            guidance = self._guide_with_heuristic()
        except Exception as e:
            print(f"An unexpected error occurred with Queen LLM: {e}. Falling back to heuristic guidance.")
            guidance = self._guide_with_heuristic()

        return guidance


def plot_grid(model, step):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1, model.width)
    ax.set_ylim(-1, model.height)

    # Plot food (green squares)
    food_x = [pos[0] for pos in model.foods]
    food_y = [pos[1] for pos in model.foods]
    ax.plot(food_x, food_y, 's', color='green', markersize=8, label='Food')

    # Plot ants (blue circles if not carrying, red if carrying)
    ant_x = [ant.pos[0] for ant in model.ants]
    ant_y = [ant.pos[1] for ant in model.ants]
    ant_colors = ['red' if ant.carrying_food else 'blue' for ant in model.ants]

    for i in range(len(ant_x)):
        ax.plot(ant_x[i], ant_y[i], 'o', color=ant_colors[i], markersize=10, alpha=0.8)

    ax.set_title(f'Ant Foraging Simulation - Step {step}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.grid(True)
    
    # Save plot instead of showing
    plt.savefig(f'ant_simulation_step_{step}.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

# Run the model and plot the grid
if __name__ == "__main__":
    # Ensure IO_API_KEY is loaded from .env for standalone execution
    if not IO_API_KEY:
        print("IO_SECRET_KEY not found in .env file. Please set it up.")
        exit()

    # Set a random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    print("Running simulation with LLM-powered ants and Queen...")
    model_llm_queen = ForagingModel(width=20, height=20, N_ants=10, N_food=20, use_queen=True, use_llm_queen=True)
    for step in range(10):
        print(f"LLM Queen Sim - Step {step+1}, Food left: {len(model_llm_queen.foods)}")
        model_llm_queen.step()
        plot_grid(model_llm_queen, f"llm_queen_step_{step}")
    print(f"LLM Queen Simulation completed! Food left: {len(model_llm_queen.foods)}")


    print("\nRunning simulation with heuristic Queen...")
    model_heuristic_queen = ForagingModel(width=20, height=20, N_ants=10, N_food=20, use_queen=True, use_llm_queen=False)
    for step in range(10):
        print(f"Heuristic Queen Sim - Step {step+1}, Food left: {len(model_heuristic_queen.foods)}")
        model_heuristic_queen.step()
        plot_grid(model_heuristic_queen, f"heuristic_queen_step_{step}")
    print(f"Heuristic Queen Simulation completed! Food left: {len(model_heuristic_queen.foods)}")


    print("\nRunning simulation without Queen (random movement)...")
    model_no_queen = ForagingModel(width=20, height=20, N_ants=10, N_food=20, use_queen=False)
    for step in range(10):
        print(f"No Queen Sim - Step {step+1}, Food left: {len(model_no_queen.foods)}")
        model_no_queen.step()
        plot_grid(model_no_queen, f"no_queen_step_{step}")
    print(f"No Queen Simulation completed! Food left: {len(model_no_queen.foods)}")

    print("\nAll simulations completed! Check the generated PNG files.")
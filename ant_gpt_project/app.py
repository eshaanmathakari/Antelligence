import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import pandas as pd
import time
import os
from dotenv import load_dotenv
import openai
import random
from random import choice
import imageio.v2 as imageio
from datetime import datetime
import json

# --- FIX: Set Matplotlib backend BEFORE importing matplotlib.pyplot ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # Now import pyplot after setting backend
# --- END FIX ---


# Load environment variables
load_dotenv()
IO_API_KEY = os.getenv("IO_SECRET_KEY")

# Page configuration
st.set_page_config(
    page_title="IO-Powered Ant Foraging Simulation",
    page_icon="🐜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1em;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1em;
        border-radius: 10px;
        margin: 0.5em 0;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🐜 IO-Powered Ant Foraging Simulation</h1>', unsafe_allow_html=True)
st.markdown("**Hackathon Project**: Autonomous Agents powered by IO Intelligence API")

# --- Class Definitions (Moved to Top) ---
class SimpleAntAgent:
    def __init__(self, unique_id, model, is_llm_controlled=True):
        self.unique_id = unique_id
        self.model = model
        self.carrying_food = False
        self.pos = (np.random.randint(model.width), np.random.randint(model.height))
        self.is_llm_controlled = is_llm_controlled
        self.api_calls = 0
        self.move_history = []
        self.food_collected_count = 0 # Track food collected by this specific ant

    def step(self, guided_pos=None):
        x, y = self.pos
        possible_steps = self.model.get_neighborhood(x, y)
        new_position = self.pos # Default to staying

        if guided_pos: # If queen provides guidance
            new_position = guided_pos
        elif self.is_llm_controlled and self.model.io_client:
            try:
                # Access prompt_style and selected_model from Streamlit session state or global scope
                # In this structure, they are global, defined in sidebar
                action = self.ask_io_for_decision(st.session_state.get('prompt_style', 'Adaptive'), st.session_state.get('selected_model', 'meta-llama/Llama-3.3-70B-Instruct'))
                self.api_calls += 1
                if action == "toward" and possible_steps:
                    # Move to adjacent cell with food if possible or just closer
                    target_food = self._find_nearest_food()
                    if target_food:
                        new_position = self._step_toward(self.pos, target_food)
                    else:
                        new_position = choice(possible_steps)
                elif action == "random" and possible_steps:
                    new_position = choice(possible_steps)
                elif action == "stay":
                    new_position = self.pos
                else: # Fallback
                    new_position = choice(possible_steps) if possible_steps else self.pos
            except Exception as e:
                #st.warning(f"API call failed for ant {self.unique_id}: {str(e)}. Falling back to random.")
                if possible_steps:
                    new_position = choice(possible_steps)
                else:
                    new_position = self.pos
        else:
            # Rule-based behavior
            if self.model.is_food_at(self.pos) and not self.carrying_food:
                new_position = self.pos # Stay to pick up food
            elif self.carrying_food and random.random() < 0.05: # Random chance to drop food at current pos
                 new_position = self.pos # Stay to drop food
            else:
                # Look for nearest food and move towards it, or move randomly
                target_food = self._find_nearest_food()
                if target_food:
                    new_position = self._step_toward(self.pos, target_food)
                else:
                    new_position = choice(possible_steps) if possible_steps else self.pos


        self.move_history.append(self.pos)
        self.pos = new_position

        # Food interaction
        if self.model.is_food_at(self.pos) and not self.carrying_food:
            self.carrying_food = True
            self.model.remove_food(self.pos)
            self.food_collected_count += 1
            if self.is_llm_controlled:
                self.model.metrics["food_collected_by_llm"] += 1
            else:
                self.model.metrics["food_collected_by_rule"] += 1

        elif self.carrying_food and random.random() < 0.1:  # Use random.random()
            self.carrying_food = False
            self.model.place_food(self.pos)

    def _find_nearest_food(self):
        if not self.model.foods:
            return None
        return min(self.model.foods,
                   key=lambda f: abs(f[0]-self.pos[0]) + abs(f[1]-self.pos[1]))

    def _step_toward(self, start, target):
        x, y = start
        tx, ty = target
        return min(self.model.get_neighborhood(x,y), key=lambda n: abs(n[0]-tx)+abs(n[1]-ty))


    def ask_io_for_decision(self, prompt_style_param, selected_model_param): # Pass these as parameters
        x, y = self.pos
        food_nearby = any(
            abs(fx - x) <= 1 and abs(fy - y) <= 1
            for fx, fy in self.model.get_food_positions()
        )

        # Refined prompt engineering for different styles
        if prompt_style_param == "Structured":
            # Direct question, expects a specific format
            prompt = (
                f"You are an ant at position ({x},{y}) on a {self.model.width}x{self.model.height} grid. "
                f"Food nearby: {food_nearby}. Carrying food: {self.carrying_food}. "
                "Considering these facts, should you 'toward' (move towards nearest food), 'random' (move randomly), or 'stay' (stay at current position)? "
                "Reply with only one word: 'toward', 'random', or 'stay'."
            )
        elif prompt_style_param == "Autonomous":
            # More open-ended, allows LLM to decide the optimal action and reasoning
            prompt = (
                f"As an autonomous ant foraging for food, my current state is: "
                f"Position: ({x},{y}), "
                f"Food available in immediate vicinity: {food_nearby}, "
                f"Am I currently carrying food: {self.carrying_food}. "
                "Based on this information, what is the single best action to take now to maximize food collection? "
                "Choose from: 'toward', 'random' (explore randomly), or 'stay' (remain stationary)."
            )
        else:  # Adaptive
            # Incorporates a simple 'efficiency' metric for adaptive behavior
            # Efficiency here is how many times the ant was at a food location
            efficiency = self.food_collected_count
            prompt = (
                f"Ant {self.unique_id} has collected {efficiency} food items so far. "
                f"Current position: ({x},{y}). "
                f"Is there food right next to me: {food_nearby}. "
                f"Am I holding food: {self.carrying_food}. "
                "Considering my past performance and current situation, what is the most optimal single word action? "
                "Options: 'toward', 'random', 'stay'."
            )

        try:
            response = self.model.io_client.chat.completions.create(
                model=selected_model_param, # Using the selected model from sidebar
                messages=[
                    {"role": "system", "content": "You are an intelligent ant foraging for food. Your response must be a single word: toward, random, or stay."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_completion_tokens=10 # Increased to give LLM a bit more room
            )
            action = response.choices[0].message.content.strip().lower()
            return action if action in ["toward", "random", "stay"] else "random" # Fallback for unexpected responses
        except openai.APICallError as e:
            #st.error(f"IO API Error for Ant {self.unique_id}: {str(e)}. Falling back to random movement.")
            return "random"
        except Exception as e:
            #st.error(f"An unexpected error occurred for Ant {self.unique_id}: {str(e)}. Falling back to random movement.")
            return "random"


class SimpleForagingModel:
    def __init__(self, width, height, N_ants, N_food,
                 agent_type="LLM-Powered", with_queen=False, use_llm_queen=False):
        self.width = width
        self.height = height
        # Ensure foods are unique
        self.foods = []
        while len(self.foods) < N_food:
            new_food_pos = (np.random.randint(width), np.random.randint(height))
            if new_food_pos not in self.foods:
                self.foods.append(new_food_pos)

        self.step_count = 0
        self.metrics = {
            "food_collected": 0,
            "total_api_calls": 0,
            "avg_response_time": 0, # Not currently implemented for live tracking
            "food_collected_by_llm": 0,
            "food_collected_by_rule": 0,
            "ants_carrying_food": 0
        }
        self.with_queen = with_queen
        self.use_llm_queen = use_llm_queen

        # Initialize IO client
        if IO_API_KEY:
            self.io_client = openai.OpenAI(
                api_key=IO_API_KEY,
                base_url="https://api.intelligence.io.solutions/api/v1/"
            )
        else:
            self.io_client = None

        # Create agents based on type
        self.ants = []
        if agent_type == "LLM-Powered":
            self.ants = [SimpleAntAgent(i, self, True) for i in range(N_ants)]
        elif agent_type == "Rule-Based":
            self.ants = [SimpleAntAgent(i, self, False) for i in range(N_ants)]
        else:  # Hybrid
            for i in range(N_ants):
                is_llm = i < N_ants // 2
                self.ants.append(SimpleAntAgent(i, self, is_llm))

        self.queen = QueenAnt(self, use_llm=self.use_llm_queen) if self.with_queen else None


    def step(self):
        self.step_count += 1
        guidance = {}
        if self.queen:
            # Pass selected_model to queen.guide()
            guidance = self.queen.guide(st.session_state.get('selected_model', 'meta-llama/Llama-3.3-70B-Instruct'))

        self.metrics["ants_carrying_food"] = 0 # Reset for current step
        for ant in self.ants:
            ant.step(guidance.get(ant))
            if ant.carrying_food:
                self.metrics["ants_carrying_food"] += 1
            if ant.is_llm_controlled:
                self.metrics["total_api_calls"] += ant.api_calls # Accumulate API calls


    def get_neighborhood(self, x, y):
        neigh = [(x+dx, y+dy)
                 for dx in (-1,0,1)
                 for dy in (-1,0,1)
                 if (dx,dy)!=(0,0)]
        # Filter out-of-bounds positions and return unique positions
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
            self.metrics["food_collected"] += 1

    def place_food(self, pos):
        if pos not in self.foods:
            self.foods.append(pos)

    def get_agent_positions(self):
        return [ant.pos for ant in self.ants]

    def get_food_positions(self):
        return self.foods


# ---------------- Queen agent ---------------- #
class QueenAnt:
    """
    One per colony. Guides worker ants. Can use LLM or a heuristic.
    """
    def __init__(self, model, use_llm=False):
        self.model = model
        self.use_llm = use_llm

    def guide(self, selected_model_param) -> dict: # Pass selected_model as parameter
        guidance = {}
        if not self.model.foods:
            return guidance

        if self.use_llm:
            return self._guide_with_llm(selected_model_param)
        else:
            return self._guide_with_heuristic()

    def _guide_with_heuristic(self) -> dict:
        guidance = {}
        ants = self.model.ants
        foods = self.model.foods
        for ant in ants:
            # pick nearest food using Manhattan distance
            if foods: # Ensure there is food to pick
                target = min(
                    foods,
                    key=lambda f: abs(f[0]-ant.pos[0]) + abs(f[1]-ant.pos[1])
                )
                # one-step move that reduces manhattan distance
                possible_moves = self.model.get_neighborhood(*ant.pos)
                if possible_moves:
                    best_step = min(
                        possible_moves,
                        key=lambda n: abs(n[0]-target[0]) + abs(n[1]-target[1])
                    )
                    guidance[ant] = best_step
                else: # No valid moves, stay put
                    guidance[ant] = ant.pos
            else: # No food, ant stays put
                guidance[ant] = ant.pos
        return guidance

    def _guide_with_llm(self, selected_model_param) -> dict: # Use selected_model_param
        guidance = {}
        if not self.model.io_client:
            st.warning("IO Client not initialized for Queen Ant. Falling back to heuristic guidance.")
            return self._guide_with_heuristic()

        # Prepare state for LLM
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

        # Prompt for the Queen LLM
        system_prompt = (
            "You are a hyper-intelligent Queen Ant. Your goal is to optimize food collection for your colony. "
            "Given the current state of ants and food, you need to provide a single best next step for each ant. "
            "For each ant, select one adjacent cell (including diagonals) or its current cell. "
            "Output a JSON object where keys are ant IDs and values are their chosen new [x, y] coordinates. "
            "Example: {\"0\": [5,6], \"1\": [10,12]}"
        )
        user_prompt = f"Current state: {json.dumps(state)}"

        llm_response_content = "" # Initialize to avoid NameError in except block
        try:
            response = self.model.io_client.chat.completions.create(
                model=selected_model_param, # Using the selected model from sidebar
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2, # Lower temperature for more deterministic guidance
                max_completion_tokens=500 # Sufficient tokens for JSON output
            )
            llm_response_content = response.choices[0].message.content.strip()
            moves = json.loads(llm_response_content)

            for ant_id_str, cell in moves.items():
                ant_id = int(ant_id_str)
                # Find the ant object
                ant_obj = next((ant for ant in self.model.ants if ant.unique_id == ant_id), None)
                if ant_obj and isinstance(cell, list) and len(cell) == 2:
                    proposed_pos = tuple(cell)
                    # Validate if proposed_pos is a valid neighbor or current pos (within bounds)
                    # This adds robustness against hallucinated coordinates from the LLM
                    valid_moves = self.model.get_neighborhood(*ant_obj.pos) + [ant_obj.pos]
                    if proposed_pos in valid_moves:
                        guidance[ant_obj] = proposed_pos
                    # If LLM suggests an invalid move, the ant will fall back to its own logic (random/toward)
                    # No explicit 'else' needed here for invalid moves, as they won't be added to guidance
        except json.JSONDecodeError as e:
            st.error(f"Queen LLM response was not valid JSON: {e}. Response: {llm_response_content}. Falling back to heuristic guidance.")
            guidance = self._guide_with_heuristic()
        except openai.APICallError as e:
            st.error(f"Queen LLM API call failed: {e}. Falling back to heuristic guidance.")
            guidance = self._guide_with_heuristic()
        except Exception as e:
            st.error(f"An unexpected error occurred with Queen LLM: {e}. Falling back to heuristic guidance.")
            guidance = self._guide_with_heuristic()

        return guidance
# --- End Class Definitions ---


# --- Sidebar configuration (Moved here and expanded) ---
st.sidebar.header("🎛️ Simulation Configuration")

with st.sidebar.expander("🌍 Environment Settings", expanded=True):
    grid_width = st.slider("Grid Width", 10, 50, 20)
    grid_height = st.slider("Grid Height", 10, 50, 20)
    n_food = st.slider("Number of Food Piles", 5, 50, 15)

with st.sidebar.expander("🐜 Ant Colony Settings", expanded=True):
    n_ants = st.slider("Number of Ants", 5, 50, 10)
    agent_type = st.selectbox("Ant Agent Type", ["LLM-Powered", "Rule-Based", "Hybrid"], index=0)

with st.sidebar.expander("🧠 LLM Settings", expanded=True):
    selected_model = st.selectbox(
        "Select LLM Model (Intelligence.io)",
        [
            "meta-llama/Llama-3.3-70B-Instruct",
            "mistral/Mistral-Nemo-Instruct-2407",
            "qwen/Qwen2-72B-Instruct",
            "Cohere/c4ai-command-r-plus"
        ],
        index=0 # Default to Llama 3.1
    )
    prompt_style = st.selectbox(
        "Ant Prompt Style",
        ["Adaptive", "Structured", "Autonomous"],
        index=0
    )

with st.sidebar.expander("👑 Queen Ant Settings", expanded=True):
    use_queen = st.checkbox("Enable Queen Overseer", value=True)
    if use_queen:
        use_llm_queen = st.checkbox("Queen uses LLM for Guidance", value=True)
    else:
        use_llm_queen = False # Ensure it's False if queen is disabled

max_steps = st.sidebar.slider("Maximum Simulation Steps", 10, 1000, 200)

# Store these values in session state for wider access if needed,
# though direct global access after definition works for most cases here.
st.session_state['grid_width'] = grid_width
st.session_state['grid_height'] = grid_height
st.session_state['n_food'] = n_food
st.session_state['n_ants'] = n_ants
st.session_state['agent_type'] = agent_type
st.session_state['selected_model'] = selected_model
st.session_state['prompt_style'] = prompt_style
st.session_state['use_queen'] = use_queen
st.session_state['use_llm_queen'] = use_llm_queen
st.session_state['max_steps'] = max_steps
# --- End Sidebar Configuration ---


# Helper function for comparison simulation
def run_comparison_simulation(params, num_steps_for_comparison=500):
    """Runs a simulation for comparison purposes and returns food collected."""
    # Ensure random seed for reproducibility in comparison
    np.random.seed(42)
    random.seed(42)

    model = SimpleForagingModel(
        width=params['grid_width'],
        height=params['grid_height'],
        N_ants=params['n_ants'],
        N_food=params['n_food'],
        agent_type=params['agent_type'],
        with_queen=params['with_queen'],
        use_llm_queen=params['use_llm_queen']
    )

    for _ in range(num_steps_for_comparison):
        if len(model.foods) == 0:
            break # Stop if all food is collected
        model.step()
    return model.metrics["food_collected"]


# Main application
def main():
    # Create columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎮 Simulation Control")

        # Control buttons
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

        with col_btn1:
            if st.button("🚀 Start Live Simulation", type="primary"):
                st.session_state.simulation_running = True
                st.session_state.current_step = 0
                # Re-initialize model to apply new parameters from sidebar
                st.session_state.model = SimpleForagingModel(
                    grid_width, grid_height, n_ants, n_food, agent_type, use_queen, use_llm_queen
                )
                st.session_state.compare_results = None # Clear comparison results


        with col_btn2:
            if st.button("⏸️ Pause Live"):
                st.session_state.simulation_running = False

        with col_btn3:
            if st.button("🔄 Reset Live"):
                st.session_state.simulation_running = False
                st.session_state.current_step = 0
                if 'model' in st.session_state:
                    del st.session_state.model # Force re-initialization
                st.session_state.compare_results = None # Clear comparison results

        with col_btn4:
            if st.button("💾 Export Data (Coming Soon)"):
                st.success("Feature coming soon!")

    with col2:
        st.subheader("📊 Live Metrics")

        # Placeholder for metrics
        if 'model' in st.session_state:
            model = st.session_state.model

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Step", model.step_count)
                st.metric("Food Collected", model.metrics["food_collected"])
                st.metric("Ants Carrying Food", model.metrics["ants_carrying_food"])


            with col_m2:
                st.metric("API Calls", model.metrics["total_api_calls"])
                st.metric("Active Ants", len(model.ants))
                st.metric("Food Left", len(model.foods))


    # Initialize simulation state
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0

    # Create model if not exists (for initial load)
    if 'model' not in st.session_state:
        st.session_state.model = SimpleForagingModel(
            grid_width, grid_height, n_ants, n_food, agent_type, use_queen, use_llm_queen
        )

    model = st.session_state.model

    # Main visualization area
    st.subheader("🗺️ Live Simulation Visualization")

    # Create visualization
    fig = go.Figure()

    # Add food items
    if model.foods:
        food_x, food_y = zip(*model.foods)
        fig.add_trace(go.Scatter(
            x=food_x, y=food_y,
            mode='markers',
            marker=dict(color='green', size=12, symbol='square'),
            name='Food',
            hovertemplate='Food at (%{x}, %{y})<extra></extra>'
        ))

    # Add ants
    if model.ants:
        ant_x, ant_y = zip(*[ant.pos for ant in model.ants])
        colors = ['red' if ant.carrying_food else 'blue' for ant in model.ants]
        types = ['LLM' if ant.is_llm_controlled else 'Rule' for ant in model.ants]

        fig.add_trace(go.Scatter(
            x=ant_x, y=ant_y,
            mode='markers',
            marker=dict(color=colors, size=10,
                        line=dict(width=1, color='DarkSlateGrey')),
            name='Ants',
            text=[f'Ant {ant.unique_id} ({types[i]})' for i, ant in enumerate(model.ants)],
            hovertemplate='%{text}<br>Position: (%{x}, %{y})<br>Carrying Food: %{marker.color}'
        ))

    # Configure layout
    fig.update_layout(
        title=f"Ant Foraging Simulation - Step {model.step_count}",
        xaxis=dict(range=[-1, grid_width], title="X Position",
                   gridcolor='lightgrey', griddash='dot'),
        yaxis=dict(range=[-1, grid_height], title="Y Position",
                   gridcolor='lightgrey', griddash='dot'),
        showlegend=True,
        width=800,
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')


    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Simulation execution
    if st.session_state.simulation_running and model.step_count < max_steps:
        # Check if there's any food left to collect
        if len(model.foods) > 0:
            with st.spinner(f"Running step {model.step_count + 1}..."):
                model.step()
                time.sleep(0.1)  # Add delay for visualization
                st.rerun()
        else:
            st.success("All food collected! Simulation completed.")
            st.session_state.simulation_running = False # Stop simulation
            st.rerun() # Refresh to show final state


    # Performance analysis
    if model.step_count > 0:
        st.subheader("📈 Performance Analysis (Live Simulation)")

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            # Food collection by agent type
            total_llm_food = model.metrics["food_collected_by_llm"]
            total_rule_food = model.metrics["food_collected_by_rule"]

            efficiency_data = pd.DataFrame({
                'Agent Type': ['LLM-Powered', 'Rule-Based'],
                'Food Collected': [total_llm_food, total_rule_food]
            })

            fig_eff = px.bar(
                efficiency_data, x='Agent Type', y='Food Collected',
                title="Food Collected by Agent Type",
                color='Agent Type',
                labels={'Food Collected': 'Food Collected'},
                text_auto=True # Display value on bars
            )
            st.plotly_chart(fig_eff, use_container_width=True)

        with col_p2:
            # API Calls Metric
            st.metric("Total API Calls (Live Sim)", model.metrics['total_api_calls'])

    st.markdown("---")
    st.subheader("📊 Comparison: Queen vs No-Queen")
    st.write("Run a separate, fixed-step simulation to compare total food collected with and without a Queen Ant.")

    if st.button("🏁 Run Comparison (Queen vs No-Queen)", type="secondary"):
        st.session_state.simulation_running = False # Stop live simulation if running
        st.session_state.compare_results = None # Clear previous comparison results

        with st.spinner("Running comparison simulations... This may take a moment."):
            # Parameters for the "No-Queen" simulation
            no_queen_params = {
                'grid_width': grid_width,
                'grid_height': grid_height,
                'n_ants': n_ants,
                'n_food': n_food,
                'agent_type': agent_type,
                'with_queen': False,
                'use_llm_queen': False # Irrelevant if no queen
            }
            food_no_queen = run_comparison_simulation(no_queen_params)

            # Parameters for the "With-Queen" simulation (using current Queen settings)
            with_queen_params = {
                'grid_width': grid_width,
                'grid_height': grid_height,
                'n_ants': n_ants,
                'n_food': n_food,
                'agent_type': agent_type,
                'with_queen': True,
                'use_llm_queen': use_llm_queen # Use the user's selected LLM queen setting
            }
            food_with_queen = run_comparison_simulation(with_queen_params)

            st.session_state.compare_results = {
                "No-Queen": food_no_queen,
                "With-Queen": food_with_queen
            }
            st.rerun() # Rerun to display results


    if 'compare_results' in st.session_state and st.session_state.compare_results is not None:
        st.write("### Comparison Results (Food Collected in 500 Steps):")
        results_df = pd.DataFrame({
            'Scenario': list(st.session_state.compare_results.keys()),
            'Food Collected': list(st.session_state.compare_results.values())
        })

        fig_compare = px.bar(
            results_df, x='Scenario', y='Food Collected',
            title="Food Collected: No-Queen vs With-Queen",
            color='Scenario',
            labels={'Food Collected': 'Total Food Collected'},
            text_auto=True # Display value on bars
        )
        st.plotly_chart(fig_compare, use_container_width=True)


    # Technical details
    with st.expander("🔧 Technical Details", expanded=False):
        st.write(f"""
        **Model Configuration:**
        - Grid Size: {grid_width} x {grid_height}
        - Agents: {n_ants} ({agent_type})
        - LLM Model: {selected_model}
        - Prompt Style: {prompt_style}
        - Queen Overseer: {'Enabled' if use_queen else 'Disabled'} ({'LLM-Powered' if use_llm_queen and use_queen else 'Heuristic' if use_queen else 'N/A'})
        - Max Steps (Live Sim): {max_steps}
        - Comparison Steps: 500

        **Current Status (Live Sim):**
        - Step: {model.step_count}/{max_steps}
        - Food Remaining: {len(model.foods)}
        - Total API Calls: {model.metrics['total_api_calls']}
        - Food Collected by LLM Ants: {model.metrics['food_collected_by_llm']}
        - Food Collected by Rule-Based Ants: {model.metrics['food_collected_by_rule']}
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🏆 <strong>Launch IO Hackathon 2025</strong> | Built with IO Intelligence API</p>
        <p>🔬 Inspired by Jimenez-Romero et al. research on LLM-powered multi-agent systems</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
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
                # Pass prompt_style and selected_model from the model's attributes
                action = self.ask_io_for_decision(self.model.prompt_style, self.model.selected_model)
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
                 agent_type="LLM-Powered", with_queen=False, use_llm_queen=False,
                 selected_model_param="meta-llama/Llama-3.3-70B-Instruct", prompt_style_param="Adaptive"):
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
        self.selected_model = selected_model_param # Store in model for ant/queen to access
        self.prompt_style = prompt_style_param # Store in model for ant to access

        # Initialize IO client
        if IO_API_KEY:
            self.io_client = openai.OpenAI(
                api_key=IO_API_KEY,
                base_url="https://api.intelligence.io.solutions/api/v1/"
            )
        else:
            self.io_client = None
            
        # initializing LLM queens report
        self.queen_llm_anomaly_rep="Queen's report will appear here when queen is active"

        # History for plotting food depletion
        self.food_depletion_history = []
        self.initial_food_count = N_food # Store initial food for 'total food remaining' calculation 


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
            guidance = self.queen.guide(self.selected_model)

        self.metrics["ants_carrying_food"] = 0 # Reset for current step
        for ant in self.ants:
            ant.step(guidance.get(ant))
            if ant.carrying_food:
                self.metrics["ants_carrying_food"] += 1
            if ant.is_llm_controlled:
                self.metrics["total_api_calls"] += ant.api_calls # Accumulate API calls

        # Calculate remaining food piles
        food_piles_remaining = len(self.foods)

        # Append to history for plotting
        self.food_depletion_history.append({
            "step": self.step_count,
            "food_piles_remaining": food_piles_remaining
        })        


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

    def guide(self, selected_model_param) -> dict:
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
            if foods:
                target = min(
                    foods,
                    key=lambda f: abs(f[0]-ant.pos[0]) + abs(f[1]-ant.pos[1])
                )
                possible_moves = self.model.get_neighborhood(*ant.pos)
                if possible_moves:
                    best_step = min(
                        possible_moves,
                        key=lambda n: abs(n[0]-target[0]) + abs(n[1]-target[1])
                    )
                    guidance[ant] = best_step
                else:
                    guidance[ant] = ant.pos
            else:
                guidance[ant] = ant.pos
        return guidance

    def _guide_with_llm(self, selected_model_param) -> dict:
        guidance = {}
        # initializing default report from Queen 
        anomaly_report_content = "Queen is evaluating colony status and anomalies..."

        if not self.model.io_client:
            st.warning("IO Client not initialized for Queen Ant. Falling back to heuristic guidance.")
            # ensuring report is available even on fallback
            self.model.queen_llm_anomaly_rep = anomaly_report_content
            return self._guide_with_heuristic()

        # ✅ Build STATE JSON
        state = {
            "ants": [
                {"id": ant.unique_id, "position": list(ant.pos), "carrying_food": ant.carrying_food, "is_llm_controlled": ant.is_llm_controlled}
                for ant in self.model.ants
            ],
            "food_positions": [list(p) for p in self.model.foods],
            "grid_size": [self.model.width, self.model.height],
            "currentstep": [self.model.step_count],
            "total_food_piles": len(self.model.foods), # number of food piles left
            "food_collected_overall": self.model.metrics["food_collected"],
            "ants_carrying_food_count": self.model.metrics["ants_carrying_food"]
        }

        # ✅ New explicit system + user prompts
        system_prompt = """
You are a hyper-intelligent Queen Ant overseeing a foraging colony.
Your primary task is to guide worker ants by proposing their next move.
Additionally, you must provide a concise "anomaly_report" based on the colony's current state and performance.
Consider factors like ant distribution, food availability, and apparent efficiency.

STRICT RULES:
- Output ONLY a single JSON object.
- This JSON object MUST have two top-level keys: "guidance" and "anomaly_report".
- "guidance" must be an object mapping ant IDs (as strings) to their chosen next positions [x,y].
  Example: {"guidance": {"0": [1,2], "1": [3,4]}, "anomaly_report": "Your observation here."}
  **VERY IMPORTANT: For each ant, the proposed next position [x,y] MUST be either the ant's current position OR an adjacent cell (horizontally, vertically, or diagonally). It CANNOT be more than one step away in any direction. All coordinates must be within the grid_size [0, width-1] and [0, height-1].**
- "anomaly_report" must be a concise string (max 2-3 sentences) with your observation. If everything seems normal, state "No significant anomalies observed."
- NO explanations, NO text, NO comments, NO markdown outside the JSON.
- If you cannot comply with the guidance format or any other rule, output exactly: {"retry": true, "anomaly_report": "Queen: Failed to generate valid guidance, attempting retry."}
"""

        user_prompt = f"""
Here is the current state of the colony at step {self.model.step_count}:

{json.dumps(state)}

Based on this information, provide the optimal guidance for each ant, and report any anomalies or key observations about the colony's foraging efficiency or unusual behavior.
"""

        llm_full_response_content = "" # Store the raw LLM response for debugging
        # Try up to 3 times to get valid JSON
        for retry_attempt in range(3):
            try:
                response = self.model.io_client.chat.completions.create(
                    model=selected_model_param,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_completion_tokens=600 # Increased to allow for longer report and more ants
                )
                llm_full_response_content = response.choices[0].message.content.strip()
                parsed_response = json.loads(llm_full_response_content)

                # Handle retry signal from LLM or if expected keys are missing
                if parsed_response.get("retry") is True:
                    anomaly_report_content = parsed_response.get("anomaly_report", f"Queen: Invalid format (retry signal), retrying. Attempt {retry_attempt + 1}.")
                    self.model.queen_llm_anomaly_rep = anomaly_report_content # Update report for UI
                    continue # Retry loop

                # Extract guidance and report from the valid JSON
                raw_guidance = parsed_response.get("guidance", {})
                anomaly_report_content = parsed_response.get("anomaly_report", "Queen: No specific report provided by LLM.")

                # Validate moves and build guidance dictionary
                valid_guidance_received = False
                for ant_id_str, cell in raw_guidance.items():
                    ant_id = int(ant_id_str)
                    ant_obj = next((ant for ant in self.model.ants if ant.unique_id == ant_id), None)
                    if ant_obj and isinstance(cell, list) and len(cell) == 2:
                        proposed_pos = tuple(cell)
                        # Ensure proposed move is within valid neighborhood or current position
                        valid_moves = self.model.get_neighborhood(*ant_obj.pos) + [ant_obj.pos]
                        if proposed_pos in valid_moves:
                            guidance[ant_obj] = proposed_pos
                            valid_guidance_received = True # At least one valid guidance received
                        else:
                            # Log or handle invalid proposed move for specific ant
                            st.info(f"Queen proposed invalid move for Ant {ant_id}: {proposed_pos}. Falling back to ant's own logic.")
                    else:
                         st.info(f"Queen provided malformed guidance for Ant {ant_id_str}: {cell}. Skipping.")


                # If LLM response was valid JSON but contained no valid guidance, we should retry or fall back
                if not raw_guidance and not valid_guidance_received and retry_attempt < 2: # Don't retry if it's the last attempt
                    anomaly_report_content = "Queen: LLM provided valid JSON but no usable guidance. Retrying."
                    self.model.queen_llm_anomaly_rep = anomaly_report_content
                    continue # Retry loop

                # Successfully parsed and (at least partially) validated. Store report and return.
                self.model.queen_llm_anomaly_rep = anomaly_report_content
                return guidance

            except json.JSONDecodeError:
                anomaly_report_content = f"Queen: JSON decoding failed. Raw response: '{llm_full_response_content[:200]}...' Retrying. Attempt {retry_attempt + 1}."
                self.model.queen_llm_anomaly_rep = anomaly_report_content
                continue # Retry loop
            except Exception as e:
                anomaly_report_content = f"Queen: Unexpected error during LLM call or parsing: {e}. Raw: '{llm_full_response_content[:200]}...' Retrying. Attempt {retry_attempt + 1}."
                self.model.queen_llm_anomaly_rep = anomaly_report_content
                continue # Retry loop

        # If all retries are exhausted
        final_fallback_report = f"Queen LLM failed after {retry_attempt + 1} retries. Falling back to heuristic. Last raw: '{llm_full_response_content[:200]}...'"
        st.warning(final_fallback_report)
        self.model.queen_llm_anomaly_rep = final_fallback_report # Ensure report is always set
        return self._guide_with_heuristic()

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
def run_comparison_simulation(params, num_steps_for_comparison=100): # Reduced steps
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
        use_llm_queen=params['use_llm_queen'],
        selected_model_param=params['selected_model'], # Pass selected_model
        prompt_style_param=params['prompt_style'] # Pass prompt_style
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
                    grid_width, grid_height, n_ants, n_food, agent_type, use_queen, use_llm_queen,
                    selected_model, prompt_style # Pass these directly
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
            grid_width, grid_height, n_ants, n_food, agent_type, use_queen, use_llm_queen,
            selected_model, prompt_style # Pass these directly
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

     # --- SECTION FOR QUEEN'S REPORT ---
    st.subheader("👑 Queen's Anomaly Report")
    if st.session_state.model.queen and st.session_state.model.use_llm_queen:
        report = st.session_state.model.queen_llm_anomaly_rep
        # Dynamic styling based on report content
        if "Failed" in report or "error" in report or "invalid" in report or "malformed" in report:
            st.error(report)
        elif "Warning:" in report or "stuck" in report or "critically low" in report or "unusual behavior" in report:
            st.warning(report)
        elif "No significant anomalies" in report or "foraging effectively" in report or "stable" in report:
            st.info(report)
        else: # Default for general observations or other states
            st.write(report)
    elif st.session_state.model.queen and not st.session_state.model.use_llm_queen:
        st.info("Queen Overseer is enabled, but not using LLM for reports (Heuristic mode).")
    else:
        st.info("Queen Overseer is disabled. Enable in sidebar to see reports.")
    # --- END QUEEN'S SECTION ---

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

        # --- FOOD RESOURCE DEPLETION ---
        if st.session_state.model.food_depletion_history: # Ensure history is not empty
            st.write("### 🌳 Food Piles Remaining Over Time") # Updated title here
            # Convert history to DataFrame
            df_food_depletion = pd.DataFrame(st.session_state.model.food_depletion_history)

            # Create the line chart
            fig_depletion = px.line(df_food_depletion,
                                    x="step",
                                    y="food_piles_remaining",
                                    title="Food Piles Remaining Over Time", # Title within the chart
                                    labels={"step": "Simulation Step", "food_piles_remaining": "Food Piles Remaining"},
                                    markers=True) # Add markers for clarity at each step

            st.plotly_chart(fig_depletion, use_container_width=True)
        # --- END OF SECTION ---    

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
                'use_llm_queen': False, # Irrelevant if no queen
                'selected_model': selected_model, # Pass selected_model
                'prompt_style': prompt_style # Pass prompt_style
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
                'use_llm_queen': use_llm_queen, # Use the user's selected LLM queen setting
                'selected_model': selected_model, # Pass selected_model
                'prompt_style': prompt_style # Pass prompt_style
            }
            food_with_queen = run_comparison_simulation(with_queen_params)

            st.session_state.compare_results = {
                "No-Queen": food_no_queen,
                "With-Queen": food_with_queen
            }
            st.rerun() # Rerun to display results


    if 'compare_results' in st.session_state and st.session_state.compare_results is not None:
        st.write("### Comparison Results (Food Collected in 100 Steps):") # Updated text
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
        - Comparison Steps: 100

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
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

# --- Class Definitions ---
class SimpleAntAgent:
    def __init__(self, unique_id, model, is_llm_controlled=True):
        self.unique_id = unique_id
        self.model = model
        self.carrying_food = False
        self.pos = (np.random.randint(model.width), np.random.randint(model.height))
        self.is_llm_controlled = is_llm_controlled
        self.api_calls = 0
        self.move_history = []
        self.food_collected_count = 0

    def step(self, guided_pos=None):
        x, y = self.pos
        possible_steps = self.model.get_neighborhood(x, y)
        new_position = self.pos

        if guided_pos and guided_pos in possible_steps + [self.pos]:
            # Queen guidance takes priority
            new_position = guided_pos
        elif self.is_llm_controlled and self.model.io_client:
            try:
                action = self.ask_io_for_decision(self.model.prompt_style, self.model.selected_model)
                self.api_calls += 1
                if action == "toward" and possible_steps:
                    target_food = self._find_nearest_food()
                    if target_food:
                        new_position = self._step_toward(self.pos, target_food)
                    else:
                        new_position = choice(possible_steps)
                elif action == "random" and possible_steps:
                    new_position = choice(possible_steps)
                elif action == "stay":
                    new_position = self.pos
                else:
                    new_position = choice(possible_steps) if possible_steps else self.pos
            except Exception as e:
                if possible_steps:
                    new_position = choice(possible_steps)
                else:
                    new_position = self.pos
        else:
            # Rule-based behavior
            if self.model.is_food_at(self.pos) and not self.carrying_food:
                new_position = self.pos  # Stay to pick up food
            elif self.carrying_food:
                # Move towards nest/home (center of grid for simplicity)
                home = (self.model.width // 2, self.model.height // 2)
                new_position = self._step_toward(self.pos, home)
            else:
                target_food = self._find_nearest_food()
                if target_food:
                    new_position = self._step_toward(self.pos, target_food)
                else:
                    new_position = choice(possible_steps) if possible_steps else self.pos

        self.move_history.append(self.pos)
        self.pos = new_position

        # FIXED: Food pickup/drop logic
        if self.model.is_food_at(self.pos) and not self.carrying_food:
            # Pick up food
            self.carrying_food = True
            self.model.remove_food(self.pos)
            self.food_collected_count += 1
            if self.is_llm_controlled:
                self.model.metrics["food_collected_by_llm"] += 1
            else:
                self.model.metrics["food_collected_by_rule"] += 1
        
        # FIXED: Only drop food at nest/home for rule-based ants, never randomly for LLM ants
        if self.carrying_food and not self.is_llm_controlled:
            home = (self.model.width // 2, self.model.height // 2)
            # Drop food if at home position or very close to it
            if abs(self.pos[0] - home[0]) <= 1 and abs(self.pos[1] - home[1]) <= 1:
                if random.random() < 0.3:  # 30% chance to drop at home
                    self.carrying_food = False
                    # Don't place food back on grid when dropping at home

    def _find_nearest_food(self):
        if not self.model.foods:
            return None
        return min(self.model.foods,
                   key=lambda f: abs(f[0]-self.pos[0]) + abs(f[1]-self.pos[1]))

    def _step_toward(self, start, target):
        x, y = start
        tx, ty = target
        possible_moves = self.model.get_neighborhood(x, y)
        if not possible_moves:
            return start
        return min(possible_moves, key=lambda n: abs(n[0]-tx)+abs(n[1]-ty))

    def ask_io_for_decision(self, prompt_style_param, selected_model_param):
        x, y = self.pos
        food_nearby = any(
            abs(fx - x) <= 2 and abs(fy - y) <= 2
            for fx, fy in self.model.get_food_positions()
        )

        if prompt_style_param == "Structured":
            prompt = (
                f"You are an ant at position ({x},{y}) on a {self.model.width}x{self.model.height} grid. "
                f"Food nearby: {food_nearby}. Carrying food: {self.carrying_food}. "
                "Should you move 'toward' food, move 'random', or 'stay'? "
                "Reply with only one word: 'toward', 'random', or 'stay'."
            )
        elif prompt_style_param == "Autonomous":
            prompt = (
                f"As an autonomous ant foraging for food, my current state is: "
                f"Position: ({x},{y}), "
                f"Food available nearby: {food_nearby}, "
                f"Currently carrying food: {self.carrying_food}. "
                "What is the best action? Choose: 'toward', 'random', or 'stay'."
            )
        else:  # Adaptive
            efficiency = self.food_collected_count
            prompt = (
                f"Ant {self.unique_id} has collected {efficiency} food items. "
                f"Current position: ({x},{y}). "
                f"Food nearby: {food_nearby}. "
                f"Carrying food: {self.carrying_food}. "
                "Best action? Options: 'toward', 'random', 'stay'."
            )

        try:
            response = self.model.io_client.chat.completions.create(
                model=selected_model_param,
                messages=[
                    {"role": "system", "content": "You are an intelligent ant. Respond with only one word: toward, random, or stay."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_completion_tokens=10
            )
            action = response.choices[0].message.content.strip().lower()
            return action if action in ["toward", "random", "stay"] else "random"
        except Exception as e:
            return "random"

class SimpleForagingModel:
    def __init__(self, width, height, N_ants, N_food,
                 agent_type="LLM-Powered", with_queen=False, use_llm_queen=False,
                 selected_model_param="meta-llama/Llama-3.3-70B-Instruct", prompt_style_param="Adaptive"):
        self.width = width
        self.height = height
        self.foods = set()
        
        # Generate unique food positions
        while len(self.foods) < N_food:
            new_food_pos = (np.random.randint(width), np.random.randint(height))
            self.foods.add(new_food_pos)

        self.step_count = 0
        self.metrics = {
            "food_collected": 0,
            "total_api_calls": 0,
            "avg_response_time": 0,
            "food_collected_by_llm": 0,
            "food_collected_by_rule": 0,
            "ants_carrying_food": 0
        }
        self.with_queen = with_queen
        self.use_llm_queen = use_llm_queen
        self.selected_model = selected_model_param
        self.prompt_style = prompt_style_param

        # Initialize IO client
        if IO_API_KEY:
            self.io_client = openai.OpenAI(
                api_key=IO_API_KEY,
                base_url="https://api.intelligence.io.solutions/api/v1/"
            )
        else:
            self.io_client = None
            
        self.queen_llm_anomaly_rep = "Queen's report will appear here when queen is active"
        self.food_depletion_history = []
        self.initial_food_count = N_food

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
            try:
                guidance = self.queen.guide(self.selected_model)
            except Exception as e:
                st.warning(f"Queen guidance failed: {e}")
                guidance = {}

        self.metrics["ants_carrying_food"] = 0
        for ant in self.ants:
            guided_pos = guidance.get(ant.unique_id)  # Use ant ID as key
            ant.step(guided_pos)
            if ant.carrying_food:
                self.metrics["ants_carrying_food"] += 1
            if ant.is_llm_controlled:
                self.metrics["total_api_calls"] += ant.api_calls

        food_piles_remaining = len(self.foods)
        self.food_depletion_history.append({
            "step": self.step_count,
            "food_piles_remaining": food_piles_remaining
        })

    def get_neighborhood(self, x, y):
        neigh = [(x+dx, y+dy)
                 for dx in (-1,0,1)
                 for dy in (-1,0,1)
                 if (dx,dy)!=(0,0)]
        valid_neigh = []
        for i,j in neigh:
            if 0 <= i < self.width and 0 <= j < self.height:
                valid_neigh.append((i,j))
        return valid_neigh

    def is_food_at(self, pos):
        return pos in self.foods

    def remove_food(self, pos):
        if pos in self.foods:
            self.foods.discard(pos)
            self.metrics["food_collected"] += 1

    def place_food(self, pos):
        if pos not in self.foods:
            self.foods.add(pos)

    def get_agent_positions(self):
        return [ant.pos for ant in self.ants]

    def get_food_positions(self):
        return list(self.foods)

# FIXED: Queen agent class
class QueenAnt:
    def __init__(self, model, use_llm=False):
        self.model = model
        self.use_llm = use_llm

    def guide(self, selected_model_param) -> dict:
        guidance = {}
        if not self.model.foods:
            self.model.queen_llm_anomaly_rep = "No food remaining for guidance"
            return guidance

        if self.use_llm:
            return self._guide_with_llm(selected_model_param)
        else:
            return self._guide_with_heuristic()

    def _guide_with_heuristic(self) -> dict:
        guidance = {}
        ants = self.model.ants
        foods = list(self.model.foods)
        
        for ant in ants:
            if foods:
                target = min(
                    foods,
                    key=lambda f: abs(f[0]-ant.pos[0]) + abs(f[1]-ant.pos[1])
                )
                possible_moves = self.model.get_neighborhood(*ant.pos) + [ant.pos]
                if possible_moves:
                    best_step = min(
                        possible_moves,
                        key=lambda n: abs(n[0]-target[0]) + abs(n[1]-target[1])
                    )
                    guidance[ant.unique_id] = best_step
        
        self.model.queen_llm_anomaly_rep = f"Heuristic guidance provided for {len(guidance)} ants"
        return guidance

    def _guide_with_llm(self, selected_model_param) -> dict:
        guidance = {}
        
        if not self.model.io_client:
            st.warning("IO Client not initialized for Queen Ant. Falling back to heuristic guidance.")
            return self._guide_with_heuristic()

        # Simplified state for LLM
        state_summary = {
            "step": self.model.step_count,
            "ants": [
                {
                    "id": ant.unique_id,
                    "pos": list(ant.pos),
                    "carrying": ant.carrying_food,
                    "type": "LLM" if ant.is_llm_controlled else "Rule"
                }
                for ant in self.model.ants
            ],
            "foods": [list(f) for f in list(self.model.foods)[:10]],  # Limit to first 10 foods
            "grid": [self.model.width, self.model.height]
        }

        # Simplified prompt
        prompt = f"""You are a Queen Ant. Guide your worker ants efficiently.

Current situation:
- Step: {self.model.step_count}
- Grid size: {self.model.width}x{self.model.height}
- Ants: {len(self.model.ants)} (some carrying food)
- Food sources: {len(self.model.foods)} remaining

For each ant, suggest the best next position (adjacent to current position or stay).
Respond ONLY with valid JSON like: {{"guidance": {{"0": [x,y], "1": [x,y]}}, "report": "brief status"}}

Ant positions and nearby food:
"""
        
        for ant in self.model.ants[:5]:  # Limit to first 5 ants to avoid token limits
            nearby_food = [f for f in self.model.foods if abs(f[0]-ant.pos[0]) <= 2 and abs(f[1]-ant.pos[1]) <= 2]
            prompt += f"Ant {ant.unique_id}: at {ant.pos}, carrying={ant.carrying_food}, nearby_food={len(nearby_food)}\n"

        try:
            response = self.model.io_client.chat.completions.create(
                model=selected_model_param,
                messages=[
                    {"role": "system", "content": "You are a Queen Ant providing tactical guidance. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_completion_tokens=300
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON from response
            try:
                # Look for JSON in the response
                if '{' in response_text and '}' in response_text:
                    start = response_text.find('{')
                    end = response_text.rfind('}') + 1
                    json_str = response_text[start:end]
                    parsed_response = json.loads(json_str)
                    
                    raw_guidance = parsed_response.get("guidance", {})
                    report = parsed_response.get("report", "Queen provided guidance")
                    
                    # Validate and convert guidance
                    for ant_id_str, pos in raw_guidance.items():
                        try:
                            ant_id = int(ant_id_str)
                            if isinstance(pos, list) and len(pos) == 2:
                                # Find the ant and validate position
                                ant = next((a for a in self.model.ants if a.unique_id == ant_id), None)
                                if ant:
                                    proposed_pos = tuple(pos)
                                    valid_moves = self.model.get_neighborhood(*ant.pos) + [ant.pos]
                                    if proposed_pos in valid_moves:
                                        guidance[ant_id] = proposed_pos
                        except (ValueError, TypeError, IndexError):
                            continue
                    
                    self.model.queen_llm_anomaly_rep = f"Queen LLM: {report} (guided {len(guidance)} ants)"
                    return guidance
                else:
                    raise json.JSONDecodeError("No JSON found", response_text, 0)
                    
            except json.JSONDecodeError:
                self.model.queen_llm_anomaly_rep = "Queen LLM: Invalid JSON response, using heuristic"
                return self._guide_with_heuristic()
                
        except Exception as e:
            self.model.queen_llm_anomaly_rep = f"Queen LLM: API error ({str(e)[:50]}), using heuristic"
            return self._guide_with_heuristic()

# --- Sidebar configuration ---
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
        index=0
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
        use_llm_queen = False

max_steps = st.sidebar.slider("Maximum Simulation Steps", 10, 1000, 200)

# Store values in session state
for key, value in [
    ('grid_width', grid_width), ('grid_height', grid_height), ('n_food', n_food),
    ('n_ants', n_ants), ('agent_type', agent_type), ('selected_model', selected_model),
    ('prompt_style', prompt_style), ('use_queen', use_queen), ('use_llm_queen', use_llm_queen),
    ('max_steps', max_steps)
]:
    st.session_state[key] = value

# Helper function for comparison simulation
def run_comparison_simulation(params, num_steps_for_comparison=100):
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
        selected_model_param=params['selected_model'],
        prompt_style_param=params['prompt_style']
    )

    for _ in range(num_steps_for_comparison):
        if len(model.foods) == 0:
            break
        model.step()
    return model.metrics["food_collected"]

# Main application
def main():
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎮 Simulation Control")
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

        with col_btn1:
            if st.button("🚀 Start Live Simulation", type="primary"):
                st.session_state.simulation_running = True
                st.session_state.current_step = 0
                st.session_state.model = SimpleForagingModel(
                    grid_width, grid_height, n_ants, n_food, agent_type, use_queen, use_llm_queen,
                    selected_model, prompt_style
                )
                st.session_state.compare_results = None

        with col_btn2:
            if st.button("⏸️ Pause Live"):
                st.session_state.simulation_running = False

        with col_btn3:
            if st.button("🔄 Reset Live"):
                st.session_state.simulation_running = False
                st.session_state.current_step = 0
                if 'model' in st.session_state:
                    del st.session_state.model
                st.session_state.compare_results = None

        with col_btn4:
            if st.button("💾 Export Data (Coming Soon)"):
                st.success("Feature coming soon!")

    with col2:
        st.subheader("📊 Live Metrics")
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

    # Create model if not exists
    if 'model' not in st.session_state:
        st.session_state.model = SimpleForagingModel(
            grid_width, grid_height, n_ants, n_food, agent_type, use_queen, use_llm_queen,
            selected_model, prompt_style
        )

    model = st.session_state.model

    # Main visualization
    st.subheader("🗺️ Live Simulation Visualization")
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
        colors = ['red' if ant.carrying_food else ('orange' if ant.is_llm_controlled else 'blue') for ant in model.ants]
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

    # Add home/nest marker
    home_x, home_y = grid_width // 2, grid_height // 2
    fig.add_trace(go.Scatter(
        x=[home_x], y=[home_y],
        mode='markers',
        marker=dict(color='purple', size=15, symbol='star'),
        name='Nest',
        hovertemplate='Nest at (%{x}, %{y})<extra></extra>'
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
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

    st.plotly_chart(fig, use_container_width=True)

    # Queen's Report Section
    st.subheader("👑 Queen's Anomaly Report")
    if model.queen and model.use_llm_queen:
        report = model.queen_llm_anomaly_rep
        if "error" in report.lower() or "failed" in report.lower():
            st.error(report)
        elif "warning" in report.lower() or "heuristic" in report.lower():
            st.warning(report)
        else:
            st.info(report)
    elif model.queen and not model.use_llm_queen:
        st.info(model.queen_llm_anomaly_rep)
    else:
        st.info("Queen Overseer disabled")

    # Simulation execution
    if st.session_state.simulation_running and model.step_count < max_steps:
        if len(model.foods) > 0:
            with st.spinner(f"Running step {model.step_count + 1}..."):
                model.step()
                time.sleep(0.1)
                st.rerun()
        else:
            st.success("All food collected! Simulation completed.")
            st.session_state.simulation_running = False
            st.rerun()

    # Performance analysis
    if model.step_count > 0:
        st.subheader("📈 Performance Analysis")
        col_p1, col_p2 = st.columns(2)

        with col_p1:
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
                text_auto=True
            )
            st.plotly_chart(fig_eff, use_container_width=True)

        with col_p2:
            if model.food_depletion_history:
                df_food_depletion = pd.DataFrame(model.food_depletion_history)
                fig_depletion = px.line(df_food_depletion,
                                        x="step",
                                        y="food_piles_remaining",
                                        title="Food Piles Over Time",
                                        markers=True)
                st.plotly_chart(fig_depletion, use_container_width=True)

    # Comparison section
    st.markdown("---")
    st.subheader("📊 Comparison: Queen vs No-Queen")

    if st.button("🏁 Run Comparison", type="secondary"):
        with st.spinner("Running comparison simulations..."):
            no_queen_params = {
                'grid_width': grid_width, 'grid_height': grid_height,
                'n_ants': n_ants, 'n_food': n_food,
                'agent_type': agent_type, 'with_queen': False,
                'use_llm_queen': False, 'selected_model': selected_model,
                'prompt_style': prompt_style
            }
            food_no_queen = run_comparison_simulation(no_queen_params)

            with_queen_params = no_queen_params.copy()
            with_queen_params['with_queen'] = True
            with_queen_params['use_llm_queen'] = use_llm_queen
            food_with_queen = run_comparison_simulation(with_queen_params)

            st.session_state.compare_results = {
                "No-Queen": food_no_queen,
                "With-Queen": food_with_queen
            }

    if 'compare_results' in st.session_state and st.session_state.compare_results:
        results_df = pd.DataFrame({
            'Scenario': list(st.session_state.compare_results.keys()),
            'Food Collected': list(st.session_state.compare_results.values())
        })

        fig_compare = px.bar(
            results_df, x='Scenario', y='Food Collected',
            title="Comparison Results (100 Steps)",
            color='Scenario',
            text_auto=True
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
        - Max Steps: {max_steps}

        **Current Status:**
        - Step: {model.step_count}/{max_steps}
        - Food Remaining: {len(model.foods)}
        - Total API Calls: {model.metrics['total_api_calls']}
        - Food Collected by LLM: {model.metrics['food_collected_by_llm']}
        - Food Collected by Rule-Based: {model.metrics['food_collected_by_rule']}
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
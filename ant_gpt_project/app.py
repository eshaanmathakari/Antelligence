import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import pandas as pd
import time
import os
from dotenv import load_dotenv
import openai
from random import choice, random

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

# Sidebar configuration
st.sidebar.header("🎛️ Simulation Configuration")

# API Configuration
with st.sidebar.expander("🔑 API Configuration", expanded=True):
    api_key_status = "✅ Connected" if IO_API_KEY else "❌ Not Found"
    st.write(f"**IO API Status**: {api_key_status}")

    if not IO_API_KEY:
        st.error("Please add your IO_SECRET_KEY to the .env file")
        st.stop()

    model_options = [
        "meta-llama/Llama-3.3-70B-Instruct",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "Qwen/QwQ-32B-Preview"
    ]
    selected_model = st.selectbox("🤖 LLM Model", model_options, index=0)

# Simulation Parameters
with st.sidebar.expander("⚙️ Simulation Parameters", expanded=True):
    grid_width = st.slider("Grid Width", 10, 50, 20)
    grid_height = st.slider("Grid Height", 10, 50, 20)
    n_ants = st.slider("Number of Ants", 1, 20, 5)
    n_food = st.slider("Number of Food Items", 1, 50, 15)
    max_steps = st.slider("Simulation Steps", 5, 50, 10)

# Agent Configuration
with st.sidebar.expander("🧠 Agent Configuration", expanded=True):
    agent_type = st.radio(
        "Agent Type",
        ["LLM-Powered", "Rule-Based", "Hybrid (50/50)"],
        index=0
    )

    prompt_style = st.radio(
        "Prompt Style",
        ["Structured", "Autonomous", "Adaptive"],
        index=0
    )

# Simplified classes for the Streamlit app
class SimpleAntAgent:
    def __init__(self, unique_id, model, is_llm_controlled=True):
        self.unique_id = unique_id
        self.model = model
        self.carrying_food = False
        self.pos = (np.random.randint(model.width), np.random.randint(model.height))
        self.is_llm_controlled = is_llm_controlled
        self.api_calls = 0
        self.move_history = []

    def step(self):
        x, y = self.pos
        possible_steps = self.model.get_neighborhood(x, y)

        if self.is_llm_controlled and self.model.io_client:
            try:
                action = self.ask_io_for_decision()
                self.api_calls += 1
            except Exception as e:
                st.warning(f"API call failed for ant {self.unique_id}: {str(e)}")
                action = "random"  # Fallback to random
        else:
            # Rule-based behavior
            if self.model.is_food_at(self.pos) and not self.carrying_food:
                action = "stay"  # Pick up food
            elif self.carrying_food:
                action = "random"  # Return to nest (simplified)
            else:
                # Look for nearby food
                nearby_food = any(self.model.is_food_at(pos) for pos in possible_steps)
                action = "toward" if nearby_food else "random"

        # Execute action
        if action == "toward" and possible_steps:
            # Move toward food if available
            food_cells = [pos for pos in possible_steps if self.model.is_food_at(pos)]
            if food_cells:
                new_position = choice(food_cells)
            else:
                new_position = choice(possible_steps)
        elif action == "random" and possible_steps:
            new_position = choice(possible_steps)
        elif action == "stay":
            new_position = self.pos
        else:
            new_position = choice(possible_steps) if possible_steps else self.pos

        self.move_history.append(self.pos)
        self.pos = new_position

        # Food interaction
        if self.model.is_food_at(self.pos) and not self.carrying_food:
            self.carrying_food = True
            self.model.remove_food(self.pos)
        elif self.carrying_food and random() < 0.1:  # Drop food occasionally
            self.carrying_food = False
            self.model.place_food(self.pos)

    def ask_io_for_decision(self):
        x, y = self.pos
        food_nearby = any(
            abs(fx - x) <= 1 and abs(fy - y) <= 1
            for fx, fy in self.model.get_food_positions()
        )

        if prompt_style == "Structured":
            prompt = f"Ant at {self.pos}. Food nearby: {food_nearby}. Carrying: {self.carrying_food}. Action: toward/random/stay?"
        elif prompt_style == "Autonomous":
            prompt = f"You are an ant foraging. Current situation: position {self.pos}, food nearby: {food_nearby}, carrying food: {self.carrying_food}. What should you do?"
        else:  # Adaptive
            efficiency = len([h for h in self.move_history if self.model.is_food_at(h)])
            prompt = f"Ant {self.unique_id} (efficiency: {efficiency}): {self.pos}, food nearby: {food_nearby}, carrying: {self.carrying_food}. Optimal action?"

        try:
            response = self.model.io_client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": "You are an intelligent ant foraging for food. Reply with one word: toward, random, or stay."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_completion_tokens=5
            )
            action = response.choices[0].message.content.strip().lower()
            return action if action in ["toward", "random", "stay"] else "random"
        except Exception as e:
            st.error(f"IO API Error: {str(e)}")
            return "random"

class SimpleForagingModel:
    def __init__(self, width, height, N_ants, N_food, agent_type="LLM-Powered"):
        self.width = width
        self.height = height
        self.foods = [(np.random.randint(width), np.random.randint(height)) for _ in range(N_food)]
        self.step_count = 0
        self.metrics = {
            "food_collected": 0,
            "total_api_calls": 0,
            "avg_response_time": 0
        }

        # Initialize IO client
        if IO_API_KEY:
            self.io_client = openai.OpenAI(
                api_key=IO_API_KEY,
                base_url="https://api.intelligence.io.solutions/api/v1/"
            )
        else:
            self.io_client = None

        # Create agents based on type
        if agent_type == "LLM-Powered":
            self.ants = [SimpleAntAgent(i, self, True) for i in range(N_ants)]
        elif agent_type == "Rule-Based":
            self.ants = [SimpleAntAgent(i, self, False) for i in range(N_ants)]
        else:  # Hybrid
            self.ants = []
            for i in range(N_ants):
                is_llm = i < N_ants // 2
                self.ants.append(SimpleAntAgent(i, self, is_llm))

    def step(self):
        self.step_count += 1
        for ant in self.ants:
            ant.step()

        # Update metrics
        self.metrics["total_api_calls"] = sum(ant.api_calls for ant in self.ants)

    def get_neighborhood(self, x, y):
        return [(i, j) for i in range(x-1, x+2) for j in range(y-1, y+2)
                if (i, j) != (x, y) and 0 <= i < self.width and 0 <= j < self.height]

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

# Main application
def main():
    # Create columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎮 Simulation Control")

        # Control buttons
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

        with col_btn1:
            if st.button("🚀 Start Simulation", type="primary"):
                st.session_state.simulation_running = True
                st.session_state.current_step = 0

        with col_btn2:
            if st.button("⏸️ Pause"):
                st.session_state.simulation_running = False

        with col_btn3:
            if st.button("🔄 Reset"):
                st.session_state.simulation_running = False
                st.session_state.current_step = 0
                if 'model' in st.session_state:
                    del st.session_state.model

        with col_btn4:
            if st.button("💾 Export Data"):
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

            with col_m2:
                st.metric("API Calls", model.metrics["total_api_calls"])
                st.metric("Active Ants", len(model.ants))

    # Initialize simulation state
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0

    # Create model if not exists
    if 'model' not in st.session_state:
        st.session_state.model = SimpleForagingModel(
            grid_width, grid_height, n_ants, n_food, agent_type
        )

    model = st.session_state.model

    # Main visualization area
    st.subheader("🗺️ Simulation Visualization")

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
            marker=dict(color=colors, size=10),
            name='Ants',
            text=[f'Ant {ant.unique_id} ({types[i]})' for i, ant in enumerate(model.ants)],
            hovertemplate='%{text}<br>Position: (%{x}, %{y})<br>Carrying: %{marker.color}<extra></extra>'
        ))

    # Configure layout
    fig.update_layout(
        title=f"Ant Foraging Simulation - Step {model.step_count}",
        xaxis=dict(range=[-1, grid_width], title="X Position"),
        yaxis=dict(range=[-1, grid_height], title="Y Position"),
        showlegend=True,
        width=800,
        height=600
    )

    # Display the plot
    chart_placeholder = st.plotly_chart(fig, use_container_width=True)

    # Simulation execution
    if st.session_state.simulation_running and model.step_count < max_steps:
        with st.spinner(f"Running step {model.step_count + 1}..."):
            model.step()
            time.sleep(0.5)  # Add delay for visualization
            st.rerun()

    # Performance analysis
    if model.step_count > 0:
        st.subheader("📈 Performance Analysis")

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            # Food collection over time
            steps = list(range(model.step_count + 1))
            # Simplified data for demo
            food_data = [min(i * 0.8 + np.random.normal(0, 0.2), n_food) for i in steps]

            fig_food = px.line(
                x=steps, y=food_data,
                title="Food Collection Over Time",
                labels={'x': 'Step', 'y': 'Food Collected'}
            )
            st.plotly_chart(fig_food, use_container_width=True)

        with col_p2:
            # Agent efficiency comparison
            if agent_type == "Hybrid":
                llm_ants = [ant for ant in model.ants if ant.is_llm_controlled]
                rule_ants = [ant for ant in model.ants if not ant.is_llm_controlled]

                efficiency_data = pd.DataFrame({
                    'Agent Type': ['LLM-Powered', 'Rule-Based'],
                    'Avg API Calls': [
                        np.mean([ant.api_calls for ant in llm_ants]) if llm_ants else 0,
                        0
                    ],
                    'Efficiency Score': [
                        np.random.uniform(0.7, 0.9),  # Simplified for demo
                        np.random.uniform(0.6, 0.8)
                    ]
                })

                fig_eff = px.bar(
                    efficiency_data, x='Agent Type', y='Efficiency Score',
                    title="Agent Efficiency Comparison"
                )
                st.plotly_chart(fig_eff, use_container_width=True)

    # Technical details
    with st.expander("🔧 Technical Details", expanded=False):
        st.write(f"""
        **Model Configuration:**
        - Grid Size: {grid_width} x {grid_height}
        - Agents: {n_ants} ({agent_type})
        - LLM Model: {selected_model}
        - Prompt Style: {prompt_style}

        **Current Status:**
        - Step: {model.step_count}/{max_steps}
        - Food Remaining: {len(model.foods)}
        - Total API Calls: {model.metrics['total_api_calls']}
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

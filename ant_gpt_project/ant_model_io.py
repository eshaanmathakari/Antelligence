"""
ant_model_io.py  –  MVP version for SBP-BRiMS Challenge 2
Only the Queen agent uses the LLM; worker ants remain light-weight.
"""

import os, json, random
from random import choice, random as rand_float
from dotenv import load_dotenv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import openai                         # already in your requirements

from queen_llm_agent import plan_moves   # <-- NEW import

# ---------------------------------------------------------------------
# 0.  ENV & global helpers
# ---------------------------------------------------------------------
load_dotenv()
IO_API_KEY = os.getenv("IO_SECRET_KEY")

def _step_toward(start, target, model):
    x, y   = start
    tx, ty = target
    return min(model.get_neighborhood(x, y) + [start],
               key=lambda n: abs(n[0]-tx) + abs(n[1]-ty))

# ---------------------------------------------------------------------
# 1.  Worker-ant behaviour (unchanged except tiny refactor)
# ---------------------------------------------------------------------
def ask_io_for_ant_decision(ant, model, io_client, selected_model):
    """
    Query IO Intelligence LLM for one-word action: toward | random | stay.
    """
    x, y = ant.pos
    food_nearby = any(abs(fx-x) <= 1 and abs(fy-y) <= 1
                      for fx, fy in model.foods)
    user_prompt = (
        f"You are an ant at {ant.pos} on a {model.width}×{model.height} grid. "
        f"Food nearby: {food_nearby}. Carrying: {ant.carrying}. "
        "Reply with one word: toward, random, or stay."
    )
    try:
        reply = io_client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system",
                 "content": "Respond ONLY with toward, random, or stay."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_completion_tokens=5,
        )
        word = reply.choices[0].message.content.strip().lower()
        return word if word in ("toward", "random", "stay") else "random"
    except Exception:
        return "random"

# ---------------------------------------------------------------------
# 2.  Core agents
# ---------------------------------------------------------------------
class AntAgent:
    def __init__(self, uid, model):
        self.unique_id   = uid
        self.model       = model
        self.carrying    = False
        self.pos         = (np.random.randint(model.width),
                            np.random.randint(model.height))

    # ----------------------------------------------------------
    def step(self, guided_pos=None):
        new_pos = self.pos
        neigh   = self.model.get_neighborhood(*self.pos)

        if guided_pos:                          # obey Queen
            new_pos = guided_pos
        else:                                   # self decision
            act = ask_io_for_ant_decision(self, self.model,
                                          self.model.io_client,
                                          self.model.selected_model)
            if act == "toward":
                target = self._nearest_food()
                new_pos = _step_toward(self.pos, target, self.model) \
                          if target else choice(neigh)
            elif act == "random":
                new_pos = choice(neigh)
            # else "stay" keeps current

        self.pos = new_pos

        # food pick-up / drop
        if self.model.is_food_at(self.pos) and not self.carrying:
            self.carrying = True
            self.model.remove_food(self.pos)
        elif self.carrying and rand_float() < 0.05:
            self.carrying = False
            self.model.place_food(self.pos)

    def _nearest_food(self):
        if not self.model.foods: return None
        return min(self.model.foods,
                   key=lambda f: abs(f[0]-self.pos[0]) + abs(f[1]-self.pos[1]))

# ----------------------------------------------------------
class QueenAnt:
    """
    Single LLM-driven planner for the colony.
    """
    def __init__(self, model, use_llm=True):
        self.model   = model
        self.use_llm = use_llm

    def guide(self):
        if not self.model.foods:
            return {}
        return self._guide_with_llm() if self.use_llm else self._guide_heuristic()

    # ----- heuristic fallback (same as before) ----------------
    def _guide_heuristic(self):
        guide = {}
        for ant in self.model.ants:
            tgt = min(self.model.foods,
                      key=lambda f: abs(f[0]-ant.pos[0]) + abs(f[1]-ant.pos[1]))
            guide[ant] = _step_toward(ant.pos, tgt, self.model)
        return guide

    # ----- NEW: delegate to plan_moves ------------------------
    def _guide_with_llm(self):
        ants_state = [{"id": a.unique_id,
                       "position": list(a.pos),
                       "carrying_food": a.carrying}
                      for a in self.model.ants]
        state = {
            "ants": ants_state,
            "food": [list(p) for p in self.model.foods],
            "size": [self.model.width, self.model.height],
        }
        moves = plan_moves(state, self.model.io_client, self.model.selected_model)
        guide = {}
        for ant in self.model.ants:
            tgt = moves.get(ant.unique_id)
            if tgt and tgt in self.model.get_neighborhood(*ant.pos) + [ant.pos]:
                guide[ant] = tgt
        return guide

# ---------------------------------------------------------------------
# 3.  Foraging model
# ---------------------------------------------------------------------
class ForagingModel:
    def __init__(self, width, height, N_ants, N_food,
                 use_queen=True, queen_uses_llm=True):
        self.width, self.height = width, height
        # unique food positions
        self.foods = []
        while len(self.foods) < N_food:
            pos = (np.random.randint(width), np.random.randint(height))
            if pos not in self.foods:
                self.foods.append(pos)

        # IO client
        self.io_client = openai.OpenAI(
            api_key=IO_API_KEY,
            base_url="https://api.intelligence.io.solutions/api/v1/"
        ) if IO_API_KEY else None
        self.selected_model = "meta-llama/Llama-3.3-70B-Instruct"

        # agents
        self.ants  = [AntAgent(i, self) for i in range(N_ants)]
        self.queen = QueenAnt(self, use_llm=queen_uses_llm) if use_queen else None

    # ----------------------------------------------------------
    def step(self):
        guidance = self.queen.guide() if self.queen else {}
        for ant in self.ants:
            ant.step(guidance.get(ant))

    # --- utility helpers ------------------------------
    def get_neighborhood(self, x, y):
        neigh = [(x+dx, y+dy)
                 for dx in (-1,0,1) for dy in (-1,0,1) if (dx, dy) != (0,0)]
        return [(i, j) for i, j in neigh
                if 0 <= i < self.width and 0 <= j < self.height]

    def is_food_at(self, pos):     return pos in self.foods
    def remove_food(self, pos):    self.foods.remove(pos)
    def place_food(self, pos):     self.foods.append(pos)

# ---------------------------------------------------------------------
# 4.  Quick CLI test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42); np.random.seed(42)
    model = ForagingModel(20, 20, 10, 20, use_queen=True, queen_uses_llm=True)
    for step in range(15):
        model.step()
        print(f"step={step:02d}  remaining_food={len(model.foods)}")

#!/usr/bin/env python
"""
queen_ant_sim.py
Run two ant-foraging simulations on the same grid:
  (1) 10 worker ants with random moves
  (2) 10 worker ants + one queen that guides them toward nearest food
After 10 ticks we compare how many food pellets are left.
"""

import random
import math
import matplotlib
matplotlib.use("Agg")  # cross‑platform backend
import matplotlib.pyplot as plt

# ----------------------  core agent types ---------------------- #
class AntAgent:
    """Worker ant that can act randomly or follow a target position."""
    def __init__(self, uid, model):
        self.id          = uid
        self.model       = model
        self.pos         = model.random_empty_cell()
        self.carrying    = False

    def step(self, guided_pos=None):
        if guided_pos:
            new_pos = guided_pos
        else:
            new_pos = random.choice(self.model.neighborhood(*self.pos))
        self.pos = new_pos

        # pickup or drop food
        if not self.carrying and self.pos in self.model.foods:
            self.carrying = True
            self.model.foods.remove(self.pos)
        elif self.carrying and random.random() < 0.05:
            self.carrying = False
            self.model.foods.add(self.pos)

class QueenAnt:
    """'LLM': looks over grid, tells each ant a best step toward nearest food."""
    def __init__(self, model):
        self.model = model

    def guide(self):
        guidance = {}
        if not self.model.foods:
            return guidance
        for ant in self.model.ants:
            target = min(
                self.model.foods,
                key=lambda f: abs(f[0]-ant.pos[0]) + abs(f[1]-ant.pos[1])
            )
            guidance[ant] = _step_toward(ant.pos, target, self.model)
        return guidance

# ----------------------  model ---------------------- #
class ForagingModel:
    def __init__(self, width=20, height=20, n_ants=10, n_food=10, use_queen=False):
        self.w, self.h = width, height
        self.foods     = set()
        self.foods.update(self.random_empty_cell() for _ in range(n_food))
        self.ants      = [AntAgent(i, self) for i in range(n_ants)]
        self.queen     = QueenAnt(self) if use_queen else None

    def random_empty_cell(self):
        while True:
            x, y = random.randrange(self.w), random.randrange(self.h)
            if (x, y) not in self.foods:
                return (x, y)

    def neighborhood(self, x, y):
        neigh = [
            (x+dx, y+dy)
            for dx in (-1,0,1)
            for dy in (-1,0,1)
            if (dx,dy)!=(0,0)
        ]
        return [(i%self.w, j%self.h) for i,j in neigh]

    def step(self):
        guidance = self.queen.guide() if self.queen else {}
        for ant in self.ants:
            ant.step(guidance.get(ant))

# ----------------------  utility ---------------------- #
def _step_toward(start, target, model):
    x, y = start
    tx, ty = target
    return min(
        model.neighborhood(x, y),
        key=lambda n: abs(n[0]-tx) + abs(n[1]-ty)
    )

# ----------------------  experiment ---------------------- #
def run_sim(use_queen=False):
    random.seed(42)
    m = ForagingModel(use_queen=use_queen)
    for _ in range(10):
        m.step()
    return len(m.foods)

def compare():
    left_no_q = run_sim(False)
    left_q    = run_sim(True)

    print("\nAfter 10 ticks with 10 ants + 10 food:")
    print(f"  • Without queen : food left = {left_no_q}")
    print(f"  • With queen    : food left = {left_q}")

    plt.bar(["No Queen", "With Queen"], [left_no_q, left_q])
    plt.ylabel("Food remaining (lower is better)")
    plt.title("Effect of Queen Guidance on Foraging Efficiency")
    plt.show()

# ----------------------  main ---------------------- #
if __name__ == "__main__":
    compare()

# --- imports -------------------------------------------------
from blockchain.client import mark_cell, record_food, mint_food
import random, numpy as np

# ... inside Ant class ---------------------------------------
def choose_move(self, grid, pheromone_map):
    """Pick next cell, biased by pheromone + food signal."""
    nbrs = self.neighbours(grid)
    scored = []
    for (nx, ny) in nbrs:
        score  = 10 if grid[nx][ny].has_food else 0
        score += pheromone_map[nx, ny] * self.cfg.trail_bias
        scored.append(((nx, ny), score))
    # highest score with epsilon-greedy
    if random.random() < self.cfg.eps_rand:
        return random.choice(nbrs)
    return max(scored, key=lambda t: t[1])[0]

def step(self, grid, pheromone_map):
    # mark visited on-chain (shared memory)
    mark_cell(self.x, self.y)

    # LLM or rule picks move ...
    nx, ny = self.choose_move(grid, pheromone_map)  # new call
    self.move_to(nx, ny)

    # deposit trail
    pheromone_map[self.x, self.y] += self.cfg.trail_strength

    # food pickup → NFT reward
    if grid[self.x][self.y].has_food and not self.carrying:
        grid[self.x][self.y].has_food = False
        self.carrying = True
        token_id = mint_food(self.wallet)          # mint NFT
        record_food(token_id, self.x, self.y)      # log on-chain

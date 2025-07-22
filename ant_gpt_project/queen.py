# queen.py  (new)
"""
Queen decentralised: simply publishes the current world snapshot
to ColonyMemory each N steps; ants read ledger + decide locally.
"""
from blockchain.client import mark_cell

class Queen:
    def __init__(self, env, every_n=5):
        self.env, self.n = env, every_n
        self.step_count = 0

    def tick(self):
        self.step_count += 1
        if self.step_count % self.n:        # every N steps
            return
        # mark nest cell so ants know it's home
        mark_cell(self.env.nest_x, self.env.nest_y)

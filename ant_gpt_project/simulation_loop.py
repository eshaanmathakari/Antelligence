import asyncio
from llm_async import batch             # new import

async def async_step(llm_ants, rule_ants, env):
    # build prompts in parallel
    prompts = [a.build_prompt(env) for a in llm_ants]
    moves   = await batch(prompts)
    for ant, act in zip(llm_ants, moves):
        ant.apply(act, env)
    for ant in rule_ants:
        ant.rule_step(env)

def run_sim(env, steps=500):
    for _ in range(steps):
        asyncio.run(async_step(env.llm_ants, env.rule_ants, env))
        env.decay_pheromones()

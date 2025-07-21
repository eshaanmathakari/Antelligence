import streamlit as st, os
from blockchain.client import w3, acct
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, json, time, random
from dotenv import load_dotenv
import openai

# ──────────────────────────────  ENV  ──────────────────────────────
load_dotenv()
IO_API_KEY = os.getenv("IO_SECRET_KEY", "")

st.set_page_config(page_title="IO Ant-Foraging ‧ Queen Thoughts",
                   page_icon="🐜", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align:center'>🐜 IO-Powered Ant Foraging Simulation</h1>",
            unsafe_allow_html=True)

# ────────────────────────  ANT  ─────────────────────────
class SimpleAntAgent:
    def __init__(self, uid, model, llm=True):
        self.uid, self.m, self.llm = uid, model, llm
        self.pos = (np.random.randint(model.w), np.random.randint(model.h))
        self.carry, self.api_calls = False, 0

    def _nbhd(self): return self.m.neigh(*self.pos)
    def _nearest_food(self):
        return min(self.m.foods,
                   key=lambda f: abs(f[0]-self.pos[0])+abs(f[1]-self.pos[1])) if self.m.foods else None
    def _step_toward(self, tgt):
        return min(self._nbhd(), key=lambda n: abs(n[0]-tgt[0])+abs(n[1]-tgt[1])) if tgt else self.pos

    def _llm_action(self):
        x,y = self.pos
        near = any(abs(fx-x)<=2 and abs(fy-y)<=2 for fx,fy in self.m.foods)
        prompt=f"Ant at ({x},{y}) carrying={self.carry}, food_nearby={near}. Word: toward/random/stay."
        rsp=self.m.io.chat.completions.create(
            model=self.m.llm_name,
            messages=[{"role":"system","content":"Return one word: toward, random, stay."},
                      {"role":"user","content":prompt}],
            temperature=0.3,max_completion_tokens=10)
        return rsp.choices[0].message.content.strip().lower()

    def step(self, guide=None):
        moves=self._nbhd(); nxt=self.pos
        if guide and guide in moves+[self.pos]:
            nxt=guide
        elif self.llm and self.m.io:
            try:
                act=self._llm_action(); self.api_calls+=1
                if act=="toward": nxt=self._step_toward(self._nearest_food()) or random.choice(moves)
                elif act=="random": nxt=random.choice(moves)
            except: nxt=random.choice(moves)
        else:  # rule
            if self.m.is_food(self.pos) and not self.carry: nxt=self.pos
            elif self.carry: nxt=self._step_toward((self.m.w//2,self.m.h//2))
            else:
                tgt=self._nearest_food(); nxt=self._step_toward(tgt) if tgt else random.choice(moves)
        self.pos=nxt

        # pick/drop
        if self.m.is_food(self.pos) and not self.carry:
            self.carry=True; self.m.foods.discard(self.pos)
            if self.llm: self.m.metrics["food_collected_by_llm"]+=1
            else: self.m.metrics["food_collected_by_rule"]+=1
        if self.carry and not self.llm:
            nest=(self.m.w//2,self.m.h//2)
            if abs(self.pos[0]-nest[0])<=1 and abs(self.pos[1]-nest[1])<=1 and random.random()<0.3:
                self.carry=False

# ────────────────────────  QUEEN  ─────────────────────────
class QueenAnt:
    def __init__(self, model, llm=True):
        self.m,self.llm=model,llm
        self.thought="Not yet acted."

    def guide(self):
        if not self.m.foods:
            self.thought="No food → no guidance."
            return {}
        return self._llm() if self.llm else self._heuristic()

    def _heuristic(self):
        g,lines={},[]
        for a in self.m.ants:
            tgt=min(self.m.foods,key=lambda f:abs(f[0]-a.pos[0])+abs(f[1]-a.pos[1]))
            step=min(self.m.neigh(*a.pos)+[a.pos],key=lambda n:abs(n[0]-tgt[0])+abs(n[1]-tgt[1]))
            g[a.uid]=step; lines.append(f"{a.uid}→{step}")
        self.thought="Heuristic | " + ", ".join(lines)
        return g

    def _llm(self):
        state={"step":self.m.step_cnt,
               "ants":[{"id":a.uid,"pos":a.pos,"carry":a.carry} for a in self.m.ants],
               "foods":list(self.m.foods)[:10],"grid":[self.m.w,self.m.h]}
        prompt=("Return JSON {'guidance':{'0':[x,y]},'report':'text'} STATE:"
                f"{json.dumps(state)}")
        try:
            rsp=self.m.io.chat.completions.create(
                model=self.m.llm_name,
                messages=[{"role":"system","content":"JSON only."},
                          {"role":"user","content":prompt}],
                temperature=0.1,max_completion_tokens=300)
            txt=rsp.choices[0].message.content.strip()
            data=json.loads(txt[txt.find("{"):txt.rfind("}")+1])
            g={int(k):tuple(v) for k,v in data["guidance"].items()}
            rep=data.get("report","(no report)")
            self.thought=f"{rep} | guided: {len(g)} ants"
            return g
        except Exception as e:
            self.thought=f"LLM fail ({e}) → heuristic"
            return self._heuristic()

# ────────────────────────  MODEL  ─────────────────────────
class ForageModel:
    def __init__(self,w,h,n_ants,n_food,agent_type,queen_on,queen_llm,llm_name):
        self.w,self.h=w,h
        self.foods=set((np.random.randint(w),np.random.randint(h)) for _ in range(n_food))
        self.io=openai.OpenAI(api_key=IO_API_KEY,
                              base_url="https://api.intelligence.io.solutions/api/v1/") if IO_API_KEY else None
        self.llm_name=llm_name
        self.metrics=dict(food_collected_by_llm=0,food_collected_by_rule=0,
                          total_api_calls=0,ants_carrying_food=0)
        self.ants=[SimpleAntAgent(i,self,
                   (agent_type=="LLM-Powered")or(agent_type=="Hybrid" and i<n_ants//2))
                   for i in range(n_ants)]
        self.queen=QueenAnt(self,queen_llm) if queen_on else None
        self.step_cnt=0
        self.food_history=[{"step":0,"food":len(self.foods)}]
        self.queen_log=[]; self.last_thought=""

    def neigh(self,x,y): 
        return[(i,j)for i in range(x-1,x+2)for j in range(y-1,y+2)
                                if(i,j)!=(x,y) and (0<=i<self.w and 0<=j<self.h)]
    def is_food(self,pos): return pos in self.foods

    def step(self):
        self.step_cnt+=1
        guidance={}
        if self.queen:
<<<<<<< Updated upstream
            guidance=self.queen.guide(); self.last_thought=self.queen.thought
            self.queen_log.append({"step":self.step_cnt,
                                   "guided":len(guidance),
                                   "thought":self.queen.thought})
            if len(self.queen_log)>5_000: self.queen_log.pop(0)

        self.metrics["ants_carrying_food"]=0
        for a in self.ants:
            a.step(guidance.get(a.uid))
            if a.carry:self.metrics["ants_carrying_food"]+=1
            self.metrics["total_api_calls"]+=a.api_calls

        self.food_history.append({"step":self.step_cnt,"food":len(self.foods)})

# ────────────────────────  SIDEBAR  ─────────────────────────
with st.sidebar:
    w=st.slider("Grid width",10,50,20)
    h=st.slider("Grid height",10,50,20)
    n_ants=st.slider("Ants",5,50,10)
    n_food=st.slider("Food",5,50,15)
    agent_type=st.selectbox("Agent type",
                            ["LLM-Powered","Rule-Based","Hybrid"])
    queen_on=st.checkbox("Enable Queen",True)
    queen_llm=st.checkbox("Queen uses LLM",True) if queen_on else False
    llm_name=st.selectbox("LLM model",
        ["meta-llama/Llama-3.3-70B-Instruct","mistral/Mistral-Nemo-Instruct-2407"])
    max_steps=st.slider("Max steps",10,1000,200)

def new_model():
    return ForageModel(w,h,n_ants,n_food,agent_type,queen_on,queen_llm,llm_name)
if "model" not in st.session_state: st.session_state.model=new_model()
M=st.session_state.model

# ─────────────────────────  CONTROL BUTTONS  ─────────────────────────
colA,colB,colC=st.columns(3)
with colA:
    if st.button("🔄 Reset"): st.session_state.model=new_model(); st.rerun()
with colB:
    if st.button("▶️ Step"): M.step(); st.rerun()
with colC:
    if st.button("⏩ Run to completion"):
        while M.step_cnt<max_steps and M.foods: M.step()
        st.rerun()

# ─────────────────────────  METRICS  ─────────────────────────
st.subheader("📊 Metrics")
m1,m2=st.columns(2)
with m1:
    st.metric("Step",M.step_cnt); st.metric("Food left",len(M.foods))
    st.metric("Ants carrying food",M.metrics["ants_carrying_food"])
with m2:
    st.metric("Food by LLM ants",M.metrics["food_collected_by_llm"])
    st.metric("Food by Rule ants",M.metrics["food_collected_by_rule"])
    st.metric("API calls",M.metrics["total_api_calls"])

# ─────────────────────────  GRID VISUAL  ─────────────────────────
st.subheader("🗺️ Live Grid")
fig=go.Figure()
if M.foods:
    fx,fy=zip(*M.foods)
    fig.add_trace(go.Scatter(x=fx,y=fy,mode="markers",
                             marker=dict(color="green",size=12,symbol="square"),
                             name="Food"))
ax,ay=zip(*[a.pos for a in M.ants])
colors=["red" if a.carry else ("orange" if a.llm else "blue") for a in M.ants]
fig.add_trace(go.Scatter(x=ax,y=ay,mode="markers",
                         marker=dict(color=colors,size=9,line=dict(width=1,color="black")),
                         name="Ants"))
fig.add_trace(go.Scatter(x=[w//2],y=[h//2],mode="markers",
                         marker=dict(color="purple",size=15,symbol="star"),name="Nest"))
fig.update_layout(xaxis_range=[-1,w],yaxis_range=[-1,h],height=500,showlegend=True)
st.plotly_chart(fig,use_container_width=True)

# ─────────────────────────  QUEEN THOUGHT  ─────────────────────────
if M.queen:
    st.subheader("👑 Queen's Latest Thought")
    st.code(M.last_thought or "(none yet)")

# ─────────────────────────  QUEEN LOG  ─────────────────────────
if M.queen:
    st.markdown("### 📜 Queen's Thought Log")
    with st.expander("📈 Summary",False):
        if M.queen_log:
            df=pd.DataFrame(M.queen_log)
            st.plotly_chart(px.line(df,x="step",y="guided",markers=True,
                                     title="Ants Guided per Step"),
                            use_container_width=True)
            st.dataframe(df[["step","guided","thought"]]
                           .sort_values("step",ascending=False).head(200),
                         height=250)
            st.download_button("💾 Download log JSON",
                               data=json.dumps(M.queen_log,indent=2),
                               file_name="queen_log.json")
        else: st.info("No thoughts yet.")

    if M.queen_log:
        step_sel=st.slider("🔍 Inspect step",1,M.queen_log[-1]["step"],
                           M.queen_log[-1]["step"])
        sel=next((d for d in M.queen_log if d["step"]==step_sel),None)
        if sel: st.code(sel["thought"])

# ─────────────────────────  PERFORMANCE ANALYSIS  ─────────────────────────
if M.step_cnt>0:
    st.subheader("📈 Performance Analysis")
    c1,c2=st.columns(2)
    with c1:
        df_e=pd.DataFrame({"Agent":["LLM","Rule"],
                           "Food":[M.metrics["food_collected_by_llm"],
                                   M.metrics["food_collected_by_rule"]]})
        st.plotly_chart(px.bar(df_e,x="Agent",y="Food",text_auto=True,
                               title="Food collected by agent type"),
                        use_container_width=True)
    with c2:
        df_f=pd.DataFrame(M.food_history)
        st.plotly_chart(px.line(df_f,x="step",y="food",
                                title="Food piles remaining",
                                markers=True),
                        use_container_width=True)

# ─────────────────────────  COMPARISON SIM  ─────────────────────────
def run_comparison(p,use_queen,use_llm):
    random.seed(42); np.random.seed(42)
    mod=ForageModel(p["w"],p["h"],p["ants"],p["food"],p["atype"],
                    use_queen,use_llm,llm_name)
    init=len(mod.foods)
    for _ in range(100):
        if not mod.foods: break
        mod.step()
    return init-len(mod.foods)

st.markdown("---")
st.subheader("📊 Comparison: Queen vs No-Queen")
if st.button("🏁 Run Comparison"):
    with st.spinner("Simulating…"):
        params=dict(w=w,h=h,ants=n_ants,food=n_food,atype=agent_type)
        collected_noq=run_comparison(params,False,False)
        collected_q =run_comparison(params,True,queen_llm)
        st.session_state.comp=dict(No_Queen=collected_noq,With_Queen=collected_q)

if "comp" in st.session_state:
    df_c=pd.DataFrame({"Scenario":list(st.session_state.comp.keys()),
                       "Food":list(st.session_state.comp.values())})
    st.plotly_chart(px.bar(df_c,x="Scenario",y="Food",text_auto=True,
                           title="Food collected in 100-step sim"),
                    use_container_width=True)

# ─────────────────────────  TECH DETAILS  ─────────────────────────
with st.expander("🔧 Technical Details"):
    st.write(f"""
**Grid**: {w}×{h}   **Ants**: {n_ants} ({agent_type})   **Food piles**: {n_food}  
**Queen**: {"ON" if queen_on else "OFF"} ({'LLM' if queen_llm and queen_on else 'Heuristic' if queen_on else '—'})  
**LLM model**: {llm_name}   **Max steps**: {max_steps}

**Current** – step {M.step_cnt} / {max_steps}, food left {len(M.foods)}, API calls {M.metrics['total_api_calls']}
""")

# ─────────────────────────  FOOTER  ─────────────────────────
st.markdown("---")
st.markdown("<div style='text-align:center'>🏆 Launch IO Hackathon 2025 – built with IO Intelligence API</div>",
            unsafe_allow_html=True)
=======
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


        # --- FORAGING EFFICIENCY MAP updates (from teammate's code) ---
        # 1. Apply decay to the entire grid at the beginning of this update phase
        self.foraging_efficiency_grid *= self.foraging_decay_rate
        # Ensure values don't go below zero after decay (optional, but good practice)
        self.foraging_efficiency_grid[self.foraging_efficiency_grid < 0.01] = 0

        # 2. Iterate through ants to add score for traversing (exploration effort)
        # We process this after all ants have moved
        for ant in self.ants:
            if ant.is_llm_controlled: # Only track LLM ants for this map
                x, y = ant.pos
                # Add score for traversing (exploration effort)
                # Ensure x, y are within bounds before accessing grid
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.foraging_efficiency_grid[x, y] += self.traverse_score_boost
        # --- END ADDITION ---

        # Apply pheromone evaporation and clipping after all ants have moved (from my previous version)
        for p_type in self.pheromone_map:
            self.pheromone_map[p_type] *= (1 - self.pheromone_decay_rate)
            self.pheromone_map[p_type] = np.clip(self.pheromone_map[p_type], 0, self.max_pheromone_value)

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

    def collect_food(self, pos, is_llm_controlled_ant):
        # This method was refactored by your teammate.
        # It now handles removing food and updating metrics for both LLM and Rule-Based ants.
        if pos in self.foods:
            self.foods.discard(pos)
            self.metrics["food_collected"] += 1

            # Update specific metrics for LLM vs Rule-Based based on who collected
            if is_llm_controlled_ant:
                self.metrics["food_collected_by_llm"] += 1
            else:
                self.metrics["food_collected_by_rule"] += 1

            # Only boost the efficiency map if an LLM-controlled ant collected food
            # Ensure pos (x, y) are within grid bounds
            x, y = pos
            if 0 <= x < self.width and 0 <= y < self.height:
                if is_llm_controlled_ant:
                    self.foraging_efficiency_grid[x, y] += self.food_collection_score_boost

    def place_food(self, pos):
        if pos not in self.foods:
            self.foods.add(pos)

    def get_agent_positions(self):
        return [ant.pos for ant in self.ants]

    def get_food_positions(self):
        return list(self.foods)

    def deposit_pheromone(self, pos, p_type, amount):
        """Deposits a specified amount of pheromone at a given position."""
        if 0 <= pos[0] < self.width and 0 <= pos[1] < self.height:
            self.pheromone_map[p_type][pos[0], pos[1]] += amount
            # Clip to max value
            self.pheromone_map[p_type][pos[0], pos[1]] = min(self.pheromone_map[p_type][pos[0], pos[1]], self.max_pheromone_value)

    def get_local_pheromones(self, pos, radius):
        """Returns the sum of pheromone levels in a given radius around a position."""
        x, y = pos
        local_trail = 0.0
        local_alarm = 0.0
        local_recruitment = 0.0

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    local_trail += self.pheromone_map['trail'][nx, ny]
                    local_alarm += self.pheromone_map['alarm'][nx, ny]
                    local_recruitment += self.pheromone_map['recruitment'][nx, ny]
        
        # Normalize by area to prevent larger radius always meaning more pheromone
        area = (2 * radius + 1)**2
        return {
            'trail': local_trail / area,
            'alarm': local_alarm / area,
            'recruitment': local_recruitment / area
        }

# Queen agent class
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

        # Summarize global pheromone information for the Queen (from my previous version)
        max_trail_val = np.max(self.model.pheromone_map['trail'])
        max_alarm_val = np.max(self.model.pheromone_map['alarm'])
        max_recruitment_val = np.max(self.model.pheromone_map['recruitment'])

        # Find approximate locations of max pheromones (for prompt brevity)
        trail_locs = np.argwhere(self.model.pheromone_map['trail'] == max_trail_val)
        alarm_locs = np.argwhere(self.model.pheromone_map['alarm'] == max_alarm_val)
        recruitment_locs = np.argwhere(self.model.pheromone_map['recruitment'] == max_recruitment_val)

        trail_pos_str = f"({trail_locs[0][0]}, {trail_locs[0][1]})" if trail_locs.size > 0 else "N/A"
        alarm_pos_str = f"({alarm_locs[0][0]}, {alarm_locs[0][1]})" if alarm_locs.size > 0 else "N/A"
        recruitment_pos_str = f"({recruitment_locs[0][0]}, {recruitment_locs[0][1]})" if recruitment_locs.size > 0 else "N/A"

        prompt = f"""You are a Queen Ant. Guide your worker ants efficiently.

Current situation:
- Step: {self.model.step_count}
- Grid size: {self.model.width}x{self.model.height}
- Ants: {len(self.model.ants)} (some carrying food)
- Food sources: {len(self.model.foods)} remaining

Pheromone Map Summary:
- Max Trail Pheromone: {max_trail_val:.2f} at {trail_pos_str} (indicates successful paths/food)
- Max Alarm Pheromone: {max_alarm_val:.2f} at {alarm_pos_str} (indicates problems/dead ends)
- Max Recruitment Pheromone: {max_recruitment_val:.2f} at {recruitment_pos_str} (indicates areas needing help or exploration)

For each ant, suggest the best next position (adjacent to current position or stay).
Consider pheromones: Encourage ants towards high trail, away from high alarm, and towards high recruitment if appropriate.
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
st.sidebar.title("Antelligence ⚙️")
st.sidebar.markdown(
    f"🔌 **IO LLM** model: `Llama-3-70b-instruct`\n\n"
    f"API endpoint: {os.getenv('IO_API_URL', 'io.net default')}"
)

# Presets
preset = st.sidebar.selectbox(
    "Load preset",
    ("Beginner Demo", "LLM Challenge", "Swarm Party"),
    help="Quick scenarios so you don't have to tweak every slider",
)

# Blockchain toggles
with st.sidebar.expander("Blockchain"):
    use_chain = st.toggle("Enable on-chain logging", value=True)
    if use_chain:
        st.write(f"📡 Connected to {w3.client_version.split('/')[0]}")
        st.write(f"🪙 Ant wallet → {acct.address[:10]}…")

# Legend
st.markdown(
    """
    **Legend:**  
    🟥 = ant carrying food | 🟦 = explorer ant | 🟢 = food | 🟡 = highest trail pheromone
    """,
    unsafe_allow_html=True,
)

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

with st.sidebar.expander("✨ Pheromone Settings", expanded=True):
    pheromone_decay_rate = st.slider("Pheromone Decay Rate", 0.01, 0.2, 0.05, 0.01)
    trail_deposit = st.slider("Trail Pheromone Deposit", 0.1, 5.0, 1.0, 0.1)
    alarm_deposit = st.slider("Alarm Pheromone Deposit", 0.1, 5.0, 2.0, 0.1)
    recruitment_deposit = st.slider("Recruitment Pheromone Deposit", 0.1, 5.0, 1.5, 0.1)
    max_pheromone_value = st.slider("Max Pheromone Value", 5.0, 20.0, 10.0, 0.5)


max_steps = st.sidebar.slider("Maximum Simulation Steps", 10, 1000, 200)

# Store values in session state
for key, value in [
    ('grid_width', grid_width), ('grid_height', grid_height), ('n_food', n_food),
    ('n_ants', n_ants), ('agent_type', agent_type), ('selected_model', selected_model),
    ('prompt_style', prompt_style), ('use_queen', use_queen), ('use_llm_queen', use_llm_queen),
    ('max_steps', max_steps),
    ('pheromone_decay_rate', pheromone_decay_rate), ('trail_deposit', trail_deposit),
    ('alarm_deposit', alarm_deposit), ('recruitment_deposit', recruitment_deposit),
    ('max_pheromone_value', max_pheromone_value)
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
    # Apply pheromone settings for comparison run
    model.pheromone_decay_rate = params['pheromone_decay_rate']
    model.trail_deposit = params['trail_deposit']
    model.alarm_deposit = params['alarm_deposit']
    model.recruitment_deposit = params['recruitment_deposit']
    model.max_pheromone_value = params['max_pheromone_value']


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
                # Apply pheromone settings from sidebar to the live model
                st.session_state.model.pheromone_decay_rate = pheromone_decay_rate
                st.session_state.model.trail_deposit = trail_deposit
                st.session_state.model.alarm_deposit = alarm_deposit
                st.session_state.model.recruitment_deposit = recruitment_deposit
                st.session_state.model.max_pheromone_value = max_pheromone_value
                
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

    # Create model if not exists (or if reset)
    if 'model' not in st.session_state:
        st.session_state.model = SimpleForagingModel(
            grid_width, grid_height, n_ants, n_food, agent_type, use_queen, use_llm_queen,
            selected_model, prompt_style
        )
        # Apply pheromone settings from sidebar to the initial model
        st.session_state.model.pheromone_decay_rate = pheromone_decay_rate
        st.session_state.model.trail_deposit = trail_deposit
        st.session_state.model.alarm_deposit = alarm_deposit
        st.session_state.model.recruitment_deposit = recruitment_deposit
        st.session_state.model.max_pheromone_value = max_pheromone_value

    model = st.session_state.model

    # Main visualization
    st.subheader("🗺️ Live Simulation Visualization")
    fig = go.Figure()

    # Add the foraging efficiency heatmap as the first trace so it's in the background
    efficiency_data = st.session_state.model.foraging_efficiency_grid
    fig.add_trace(go.Heatmap(
        z=efficiency_data.T, # Transpose for correct orientation (x=cols, y=rows)
        x=np.arange(model.width),
        y=np.arange(model.height),
        colorscale='YlOrRd', # A good, visible hot-spot color scale
        colorbar=dict(title='Efficiency Score'),
        opacity=0.5, # Make it semi-transparent so ants/food are visible
        hoverinfo='skip', # Don't show hover info for heatmap cells
        name='LLM Foraging Hotspot', # Name for legend
        zmin=0, # Minimum value for color scale
        zmax=np.max(efficiency_data) * 1.2 if np.max(efficiency_data) > 0 else 1 # Scale max dynamically for visual effect, handle zero case
    ))


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

    # Pheromone Map Visualizations (from my previous version)
    st.subheader("🧪 Pheromone Maps")
    
    pheromone_cols = st.columns(3)

    # Trail Pheromone Heatmap
    with pheromone_cols[0]:
        st.markdown("##### Trail Pheromone")
        fig_trail = go.Figure(data=go.Heatmap(
            z=model.pheromone_map['trail'].T, # Transpose for correct orientation
            x=list(range(model.width)),
            y=list(range(model.height)),
            colorscale='Greens',
            zmin=0, zmax=model.max_pheromone_value
        ))
        fig_trail.update_layout(
            xaxis=dict(title="X"), yaxis=dict(title="Y"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=300
        )
        st.plotly_chart(fig_trail, use_container_width=True)

    # Alarm Pheromone Heatmap
    with pheromone_cols[1]:
        st.markdown("##### Alarm Pheromone")
        fig_alarm = go.Figure(data=go.Heatmap(
            z=model.pheromone_map['alarm'].T, # Transpose for correct orientation
            x=list(range(model.width)),
            y=list(range(model.height)),
            colorscale='Reds',
            zmin=0, zmax=model.max_pheromone_value
        ))
        fig_alarm.update_layout(
            xaxis=dict(title="X"), yaxis=dict(title="Y"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=300
        )
        st.plotly_chart(fig_alarm, use_container_width=True)

    # Recruitment Pheromone Heatmap
    with pheromone_cols[2]:
        st.markdown("##### Recruitment Pheromone")
        fig_recruitment = go.Figure(data=go.Heatmap(
            z=model.pheromone_map['recruitment'].T, # Transpose for correct orientation
            x=list(range(model.width)),
            y=list(range(model.height)),
            colorscale='Blues',
            zmin=0, zmax=model.max_pheromone_value
        ))
        fig_recruitment.update_layout(
            xaxis=dict(title="X"), yaxis=dict(title="Y"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=300
        )
        st.plotly_chart(fig_recruitment, use_container_width=True)


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

        # Teammate's commented out Foraging Efficiency Map visualization (kept commented)
        #st.write("### 🌐 Foraging Efficiency Map (LLM Activity)")
        #efficiency_data = st.session_state.model.foraging_efficiency_grid
        #fig_efficiency = px.imshow(efficiency_data.T, # Transpose for correct orientation (x=cols, y=rows)
        #                            color_continuous_scale="Hot", # Use a "hot" color scale
        #                            labels=dict(x="X Position", y="Y Position", color="Efficiency Score"),
        #                            title="LLM Foraging Activity Hotspot",
        #                            origin="lower", # Important for correct Y-axis orientation
        #                            range_color=[0, np.max(efficiency_data) * 1.2]) # Scale color range dynamically
        #fig_efficiency.update_xaxes(side="top", showgrid=False, zeroline=False,
        #                            tickvals=np.arange(model.width), ticktext=np.arange(model.width))
        #fig_efficiency.update_yaxes(showgrid=False, zeroline=False,
        #                            tickvals=np.arange(model.height), ticktext=np.arange(model.height))
        #st.plotly_chart(fig_efficiency, use_container_width=True)

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
                'prompt_style': prompt_style,
                'pheromone_decay_rate': pheromone_decay_rate, # Pass pheromone params
                'trail_deposit': trail_deposit,
                'alarm_deposit': alarm_deposit,
                'recruitment_deposit': recruitment_deposit,
                'max_pheromone_value': max_pheromone_value
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

        **Pheromone Configuration:**
        - Decay Rate: {pheromone_decay_rate}
        - Trail Deposit: {trail_deposit}
        - Alarm Deposit: {alarm_deposit}
        - Recruitment Deposit: {recruitment_deposit}
        - Max Pheromone Value: {max_pheromone_value}

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
>>>>>>> Stashed changes

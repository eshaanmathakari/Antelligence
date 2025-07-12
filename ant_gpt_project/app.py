import streamlit as st
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

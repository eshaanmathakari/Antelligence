import os, json, functools
from pathlib import Path
from web3 import Web3
from eth_account import Account

_RPC      = os.getenv("CHAIN_RPC", "http://127.0.0.1:8545")
_PRIV_KEY = os.getenv("PRIVATE_KEY")                # local dev key
_FOOD     = os.getenv("FOOD_ADDR")                  # deployed FoodToken
_MEMO     = os.getenv("MEMORY_ADDR")                # deployed ColonyMemory

w3   = Web3(Web3.HTTPProvider(_RPC))
acct = Account.from_key(_PRIV_KEY)
nonce = functools.partial(w3.eth.get_transaction_count, acct.address)

def _load(name):
    root = Path(__file__).resolve().parent / "artifacts" / "contracts"
    file = root / f"{name}.sol" / f"{name}.json"
    return json.loads(file.read_text())["abi"]

food  = w3.eth.contract(address=_FOOD,   abi=_load("FoodToken"))
mem   = w3.eth.contract(address=_MEMO,   abi=_load("ColonyMemory"))

def _tx(fn):
    tx = fn.build_transaction({
        "from":  acct.address,
        "nonce": nonce(),
        "gas":   250_000,
        "gasPrice": w3.to_wei("2", "gwei"),
    })
    signed = acct.sign_transaction(tx)
    return w3.eth.send_raw_transaction(signed.rawTransaction)

# ------------- helpers exposed to simulation ----------------
def mark_cell(x:int, y:int):
    _tx(mem.functions.markVisited(x, y))

def record_food(token_id:int, x:int, y:int):
    _tx(mem.functions.recordFood(token_id, x, y))

def mint_food(receiver:str):
    _tx(food.functions.mint(receiver))

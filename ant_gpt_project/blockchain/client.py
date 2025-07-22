import os
from dotenv import load_dotenv
from web3 import Web3
from eth_account import Account
import json # Import json to parse ABI if needed, though we'll assume it's a string from .env for now

# Load environment variables from .env file
load_dotenv()

# Get RPC URL and Private Key from environment variables
# Prioritize local chain RPC if available, otherwise fall back to Sepolia
_CHAIN_RPC = os.getenv("CHAIN_RPC")
_SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL")

# Choose which RPC URL to use
RPC_URL = None
if _CHAIN_RPC and _CHAIN_RPC != "http://127.0.0.1:8545": # Check if local RPC is set and not just default placeholder
    RPC_URL = _CHAIN_RPC
    print(f"Using local RPC: {RPC_URL}")
elif _SEPOLIA_RPC_URL:
    RPC_URL = _SEPOLIA_RPC_URL
    print(f"Using Sepolia RPC: {RPC_URL}")
else:
    raise ValueError("Neither CHAIN_RPC nor SEPOLIA_RPC_URL is set in .env. Please configure at least one.")


_PRIV_KEY = os.getenv("PRIVATE_KEY")

# Check if private key is loaded and not empty
if not _PRIV_KEY:
    raise ValueError("PRIVATE_KEY environment variable not set or is empty. Please set it in your .env file with a valid testnet private key.")

# Initialize Web3 provider
w3 = Web3(Web3.HTTPProvider(RPC_URL))

# Check connection
if not w3.is_connected():
    raise ConnectionError(f"Failed to connect to Ethereum node at {RPC_URL}. Please check your RPC URL and network connection.")
else:
    print(f"Successfully connected to Ethereum node at {RPC_URL}")
    print(f"Current Chain ID: {w3.eth.chain_id}")

# Load account from private key
try:
    acct = Account.from_key(_PRIV_KEY)
    print(f"Account loaded: {acct.address}")
    # Verify that the account has some balance (optional but good for debugging)
    balance_wei = w3.eth.get_balance(acct.address)
    balance_eth = w3.from_wei(balance_wei, 'ether')
    print(f"Account balance: {balance_eth:.4f} ETH")
    if balance_eth == 0:
        print("WARNING: Account has 0 ETH. Transactions will likely fail. Please fund it via a faucet.")
except Exception as e:
    raise ValueError(f"Failed to load account from private key: {e}. Ensure PRIVATE_KEY is a valid hex string (e.g., '0x...' or without '0x' prefix if it's just the hex string).")

# Load contract addresses from environment variables
FOOD_CONTRACT_ADDRESS = os.getenv("FOOD_ADDR")
MEMORY_CONTRACT_ADDRESS = os.getenv("MEMORY_ADDR")

if not FOOD_CONTRACT_ADDRESS:
    print("WARNING: FOOD_ADDR not set in .env. Food contract interactions may fail.")
if not MEMORY_CONTRACT_ADDRESS:
    print("WARNING: MEMORY_ADDR not set in .env. Memory contract interactions may fail.")

# You can define a placeholder ABI here if you don't want to load it from Streamlit
# For demonstration, you might want to load it from a separate JSON file or hardcode it
# Example:
# with open('path/to/your/FoodContract.json', 'r') as f:
#     FOOD_CONTRACT_ABI = json.load(f)['abi']
# Or if you want to use the one from app.py's sidebar:
# FOOD_CONTRACT_ABI = None # Will be passed from Streamlit's session_state

# The `w3` and `acct` objects are now ready to be imported and used by `app.py`

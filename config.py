import json

abi_file = "abi/FederatedLearning.json"

# Load the contract's ABI
with open(abi_file) as f:
    CONTRACT_ABI = json.load(f)

CONTRACT_ADDRESS ="0x5FbDB2315678afecb367f032d93F642f64180aa3"
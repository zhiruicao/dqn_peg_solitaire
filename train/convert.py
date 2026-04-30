import torch
import json
from network import DQN

model = DQN()
    
state_dict = torch.load('model.pth', map_location='cpu')
    
if 'state_dict' in state_dict:
    model.load_state_dict(state_dict['state_dict'])
else:
    model.load_state_dict(state_dict)
    
model.eval()
    
weights_dict = {}
for name, param in model.named_parameters():
    weights_dict[name] = param.detach().numpy().tolist()

with open('weights.json', 'w') as f:
    json.dump(weights_dict, f)
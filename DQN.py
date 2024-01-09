import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
class DQN(nn.Module):
    
    def __init__(self, env, learning_rate=1e-3, neurons=16):
        super(DQN, self).__init__()
        """
        Params
        ======
        n_inputs: state space size 
        n_outputs: action space size 
        actions: possible actions array
        device: cpu or cuda
        model: neural network definition
        """
        version = "third"
        self.input_shape =  env.observation_space 
        self.n_outputs = 6 
        self.actions =  np.arange(self.n_outputs)
      
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if version == "first":
            ### Neural network definition
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.input_shape, neurons, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(neurons, neurons, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(neurons, self.n_outputs, bias=True)).to(device=self.device)
        elif version == "second":
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.input_shape, neurons, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(neurons, neurons, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(neurons, neurons, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(neurons, neurons, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(neurons, self.n_outputs, bias=True)).to(device=self.device)
        elif version == "third":
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.input_shape, neurons, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(neurons, neurons, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(neurons, neurons, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(neurons, neurons, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(neurons, neurons, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(neurons, neurons, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(neurons, self.n_outputs, bias=True)).to(device=self.device)


       
        ## Optimizer initialization
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    ### e-greedy method
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)
        else:
            qvals = self.get_qvals(state)
            action= torch.max(qvals, dim=-1)[1].item()
        return action


    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array(state)
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.model(state_t)
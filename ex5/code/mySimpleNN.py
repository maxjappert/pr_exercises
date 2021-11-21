import torch


class mySimpleNN(torch.nn.Module):
    '''
    Define the Neural network architecture
    '''

    def __init__(self):
        super(mySimpleNN, self).__init__()
    # TODO: Define a simple neural network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    # TODO: Define the network forward propagation from x -> y_hat
        y_hat = None
        return y_hat


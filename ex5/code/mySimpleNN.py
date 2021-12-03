import torch


class mySimpleNN(torch.nn.Module):
    '''
    Define the Neural network architecture
    '''

    def __init__(self):
        super(mySimpleNN, self).__init__()
    # TODO: Define a simple neural network

        #self.linear = torch.nn.Linear(2, 1, bias=True)  # bias is default True

        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Sigmoid(),
            torch.nn.Linear(2, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 1),
            torch.nn.Sigmoid()
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
    # TODO: Define the network forward propagation from x -> y_hat
        #y_hat = self.linear.forward(x)

        y_hat = self.linear_relu_stack(x)

        return y_hat


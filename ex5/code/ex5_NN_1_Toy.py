import sys
import torch


def toyNetwork() -> None:
    # TODO: Implement network as given in the exercise sheet.
    # Manual implementation functionality when computing loss, gradients and optimization
    # i.e. do not use torch.optim or any of the torch.nn functionality
    # Torch documentation: https://pytorch.org/docs/stable/index.html

    # TODO: Define weight variables using: torch.tensor([], requires_grad=True)
    # TODO: Define data: x, y using torch.tensor
    # TODO: Define learning rate

    # TODO: Train network until convergence
    # TODO: Define network forward pass connectivity
    # TODO: Get gradients of weights and manually update the network weights

    # Steps:
    # 1 - compute error
    # 2 - do backward propagation, use: error.backward() to do so
    # 3 - update weight variables according to gradient and learning rate
    # 4 - Zero weight gradients with w_.grad_zero_()

    # define the network structure (2 input nodes, 2 hidden layer nodes and 1 output node)
    n_input, n_hidden, n_output = 2, 2, 1

    # weights
    w1 = torch.tensor([0.7, 1.3, 1.5, 0.1], requires_grad=True)
    w2 = torch.tensor([0.7, 0.8], requires_grad=True)

    X = torch.tensor(([1, 1]), dtype=torch.float)
    Y = torch.tensor(([1]), dtype=torch.float)

    learning_rate = 0.2

    def forward(x):
        h = torch.matmul(x, w1)
        h_out = sigmoid(h)
        y_hat = torch.matmul(h_out, w2)  # do not have to squash it, because linear function for output

    def sigmoid(s):
        return 1 / (1 + torch.exp(-s))


if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("Neural network toy example!")
    toyNetwork()
    print("Done!")

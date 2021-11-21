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



if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("Neural network toy example!")
    toyNetwork()
    print("Done!")

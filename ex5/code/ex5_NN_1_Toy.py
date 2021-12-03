import sys
import torch


def toyNetwork() -> None:
    # Implement network as given in the exercise sheet.
    # Manual implementation functionality when computing loss, gradients and optimization
    # i.e. do not use torch.optim or any of the torch.nn functionality
    # Torch documentation: https://pytorch.org/docs/stable/index.html

    # Define weight variables using: torch.tensor([], requires_grad=True)
    # Define data: x, y using torch.tensor
    # Define learning rate

    # Train network until convergence
    # Define network forward pass connectivity
    # Get gradients of weights and manually update the network weights

    # Steps:
    # 1 - compute error
    # 2 - do backward propagation, use: error.backward() to do so
    # 3 - update weight variables according to gradient and learning rate
    # 4 - Zero weight gradients with w_.grad_zero_()

    # define the network structure (2 input nodes, 2 hidden layer nodes and 1 output node)
    n_input, n_hidden, n_output = 2, 2, 1

    # we define the tensors
    x = torch.tensor([1.0, 1.0], requires_grad=True)
    y = torch.tensor(1.0, requires_grad=True)

    w = torch.tensor([[0.7, 1.5],
                      [1.3, 0.1]], requires_grad=True)

    w2 = torch.tensor([0.7, 0.8], requires_grad=True)

    wb = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)

    for i in range(0, 10):
        # since the bias is always b = 1, it corresponds to its weight, since w = 1 * w
        bias = torch.tensor([wb[0], wb[1]])

        h = sigmoid(w @ x + bias)

        y_hat = torch.sum(h * w2) + wb[2]

        error = 0.5 * torch.pow(y - y_hat, 2)

        error.backward()  # Compute the Gradients for w and b (requires_grad=True)

        lr = 1

        # from the deep learning notebook
        with torch.no_grad():  # Temporarily set all requires_grad=False
            w -= lr * w.grad
            w2 -= lr * w2.grad
            wb -= lr * wb.grad
            # Remember to zero the gradients!
            # If not, the gradients will be accumulated
            w.grad.zero_()
            w2.grad.zero_()
            wb.grad.zero_()

        print("Error: {:.4f}".format(error))

def sigmoid(s):
    return 1 / (1 + torch.exp(-s))

if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("Neural network toy example!")
    toyNetwork()
    print("Done!")

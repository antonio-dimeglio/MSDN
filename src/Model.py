import torch.nn as nn 

class MSDN(nn.Module):
    """
        Model class for the MSDN model.
    """
    def __init__(self):
        """
            Constructor for the MSDN model class.
        """
        super(MSDN, self).__init__()

        # The inputs are (for now), images of shape (4, 256, 256)
        # For now no convolutions are used, just a simple linear layer, for testing purposes
        # The network is supposed to return a reconstructed phantom of the sam shape as the input sinogram
        self.fc1 = nn.Linear(4 * 256 * 256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4 * 256 * 256)

    def forward(self, x):
        """
            Forward pass of the model.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor.
        """
        x = x.view(-1, 4 * 256 * 256)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.view(-1, 4, 256, 256)

        return x
        
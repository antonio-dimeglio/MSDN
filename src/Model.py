import torch
import torch.nn as nn 
import torch.nn.functional as F


class MSDLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(MSDLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    

class MSDBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(MSDBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = MSDLayer(in_channels + i * growth_rate, growth_rate, dilation=2**i)
            self.layers.append(layer)           

    def forward(self, x):
        features = [x] 
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)
    


class MSDN(nn.Module):
    """
        Model class for the MSDN model.
    """
    def __init__(self, in_channels, num_classes, growth_rate=16, num_blocks=5, num_layers_per_block=4):
        """
            Constructor for the MSDN model class.
        """
        super(MSDN, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()
        in_channels = growth_rate
        for _ in range(num_blocks):
            block = MSDBlock(in_channels, growth_rate, num_layers_per_block)
            self.blocks.append(block)
            in_channels += growth_rate * num_layers_per_block
        self.final_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)


    def forward(self, x):
        """
            Forward pass of the model.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor.
        """
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x
        

def main():
    model = MSDN(in_channels=3, num_classes=10)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)

if __name__ == '__main__':
    main()
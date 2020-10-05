import torch.nn as nn


def initialize_weights(m, activation):

    for module in m.modules():
        module_name = module.__class__.__name__

        if activation not in ('relu', 'leaky_relu'):
            raise Exception('Please specify your activation function name')
        if module_name.find('Conv2') != -1:
            nn.init.kaiming_uniform_(module.weight, nonlinearity=activation)
        elif module_name.find('BatchNorm') != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0.0)
        elif module_name.find('Linear') != -1:
            nn.init.kaiming_uniform_(module.weight, nonlinearity=activation)
            if module.bias is not None:
                module.bias.data.fill_(0.1)
        else:
            print('Cannot initialize the layer :', module_name)


class Flatten(nn.Module):
    def forward(self, x):
        """
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        """
        batch_size = x.size(0)
        out = x.view(batch_size, -1)
        return out  # (batch_size, *size)
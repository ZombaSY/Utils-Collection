import torch.nn as nn


def initialize_weights(m, activation):
    classname = m.__class__.__name__

    if activation not in ('relu', 'leaky_relu'):
        raise Exception('Please specify your activation function name')
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(m.weight, nonlinearity=activation)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight, nonlinearity=activation)
        if m.bias is not None:
            m.bias.data.fill_(0.1)
    else:
        pass


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
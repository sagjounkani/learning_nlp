import torch
from torch import nn
from typing import Optional, List
from collections import defaultdict


def assert_shape(x, shape: list):
    """ ex: assert_shape(conv_input_array, [8, 3, None, None]) """
    assert len(x.shape) == len(shape), (x.shape, shape)
    for _a, _b in zip(x.shape, shape):
        if isinstance(_b, int):
            assert _a == _b, (x.shape, shape)


def assert_shapes(x, x_shape, y, y_shape):
    assert_shape(x, x_shape)
    assert_shape(y, y_shape)

    shapes = defaultdict(set)
    for arr, shape in [(x, x_shape), (y, y_shape)]:
        for i, char in enumerate(shape):
            if isinstance(char, str):
                shapes[char].add(arr.shape[i])
    for _, _set in shapes.items():
        assert len(_set) == 1, (x, x_shape, y, y_shape)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear, self).__init__()
        self.init = nn.init.xavier_uniform_
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.w = nn.parameter.Parameter(self.init(torch.rand((self.out_features, self.in_features))))
        if self.bias:
            self.b = nn.parameter.Parameter(torch.rand(self.out_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert isinstance(input, torch.Tensor)
        if self.bias:
            return input.matmul(self.w.T).add(self.b)
        return input.matmul(self.w.T)


class RNNLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, nonlinearity: str = 'tanh', bias: bool = True):
        super(RNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if nonlinearity == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == 'relu':
            self.nonlinearity = nn.ReLU()
        self.i2h = Linear(in_features=self.input_size, out_features=self.hidden_size, bias=self.bias)
        self.h2h = Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=self.bias)

    def forward(self, input: torch.Tensor, h_0: Optional[torch.Tensor] = None):
        if h_0 is None:
            h_0 = nn.parameter.Parameter(torch.zeros((1, self.hidden_size)))
        output = self.nonlinearity(self.i2h(input).add(self.h2h(h_0)))
        return output


class LSTMLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, nonlinearity: str = 'tanh', bias: bool = True):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if nonlinearity == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == 'relu':
            self.nonlinearity = nn.ReLU()
        self.i2h = Linear(in_features=self.input_size, out_features=self.hidden_size, bias=self.bias)
        self.h2h = Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=self.bias)

    def forward(self, input: torch.Tensor, h_0: Optional[torch.Tensor] = None):
        if h_0 is None:
            h_0 = nn.parameter.Parameter(torch.zeros((1, self.hidden_size)))
        output = self.nonlinearity(self.i2h(input).add(self.h2h(h_0)))
        return output


class GRULayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, nonlinearity: str = 'tanh', bias: bool = True):
        super(GRULayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if nonlinearity == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == 'relu':
            self.nonlinearity = nn.ReLU()
        self.i2h = Linear(in_features=self.input_size, out_features=self.hidden_size, bias=self.bias)
        self.h2h = Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=self.bias)

    def forward(self, input: torch.Tensor, h_0: Optional[torch.Tensor] = None):
        if h_0 is None:
            h_0 = nn.parameter.Parameter(torch.zeros((1, self.hidden_size)))
        output = self.nonlinearity(self.i2h(input).add(self.h2h(h_0)))
        return output


class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1, nonlinearity: str = 'tanh',
                 bias: bool = True, bidirectional: bool = False):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity
        self.layers: List[List[RNNLayer]] = [self.__generate_layer()]
        for i in range(1, self.num_layers):
            self.layers.append(self.__generate_layer())
        self.num_directions = 2 if self.bidirectional else 1

    def __generate_layer(self):
        if not self.bidirectional:
            return [self.__forward_layer()]
        else:
            return [self.__forward_layer(), self.__backward_layer()]

    def __forward_layer(self):
        return RNNLayer(input_size=self.input_size,
                        hidden_size=self.hidden_size,
                        nonlinearity=self.nonlinearity,
                        bias=self.bias)

    def __backward_layer(self):
        return RNNLayer(input_size=self.input_size,
                        hidden_size=self.hidden_size,
                        nonlinearity=self.nonlinearity,
                        bias=self.bias)

    def __forward_unidirectional(self, input: torch.Tensor, h_0: Optional[torch.Tensor] = None, layer_direction: int = 0):
        sequence_length = input.shape[0]

        if not h_0:
            h_0 = nn.parameter.Parameter(torch.zeros((self.num_layers, self.hidden_size)))

        for step in torch.arange(0, sequence_length, dtype=torch.int32):
            for layer in self.layers:
                h_0 = layer[layer_direction](torch.index_select(input, 0, step), h_0)
            output.append(h_0)

        return torch.cat(output[1:], dim=0), output[-1]

    def __forward_bidirectional(self, input: torch.Tensor, h_0: Optional[torch.Tensor] = None):
        flipped_input = torch.flip(input, dims=[0])
        output_f, h_n_f = self.__forward_unidirectional(input, h_0)
        output_b, h_n_b = self.__forward_unidirectional(flipped_input, h_0, layer_direction=1)
        return torch.cat([output_f, output_b], dim=1), torch.cat([h_n_f, h_n_b], dim=1)

    def forward(self, input: torch.Tensor, h_0: Optional[torch.Tensor] = None):

        assert_shape(input, [None, self.input_size])
        if not h_0:
            assert_shape(h_0, [self.num_directions * self.num_layers, self.hidden_size])

        if not self.bidirectional:
            output, h_n = self.__forward_unidirectional(input, h_0)
        else:
            output, h_n = self.__forward_bidirectional(input, h_0)
        return output, h_n


if __name__ == '__main__':
    rnn = RNN(input_size=20, bidirectional=True, num_layers=2)
    rnn(torch.ones((2, 20)))

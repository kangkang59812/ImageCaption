import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy


class aLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, attrhidden_size):
        super(aLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attrhidden_size = attrhidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.weight_ah = Parameter(torch.randn(4 * hidden_size, attrhidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        self.bias_ah = Parameter(torch.randn(4 * hidden_size))

    def forward(self, input, state, attr_hidden):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh +
                 torch.mm(attr_hidden, self.weight_ah.t()) + self.bias_ah)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy


if __name__ == "__main__":
    lstm_cell = torch.nn.LSTMCell(5, 10)
    hidden_size = 10
    input_size = 5
    a = torch.randn(2, 10)
    b = torch.tensor(a)
    q = torch.randn(4 * hidden_size, hidden_size)
    w = torch.randn(4 * hidden_size, input_size)
    e = torch.randn(4 * hidden_size)
    r = torch.randn(4 * hidden_size)
    lstm_cell.weight_hh.data = q
    lstm_cell.weight_ih.data = w
    lstm_cell.bias_hh.data = e
    lstm_cell.bias_ih.data = r

    input = torch.randn(2, 5)
    attr = torch.randn(2, 1024)
    h_0 = a
    c_0 = b
    h1, c1 = lstm_cell(input, (h_0, c_0))
    print(h1, c1)

    lstm_cell2 = aLSTMCell(5, 10, 1024)

    lstm_cell2.weight_hh.data = q
    lstm_cell2.weight_ih.data = w
    lstm_cell2.bias_hh.data = e
    lstm_cell2.bias_ih.data = torch.ones(4 * hidden_size)

    h_02 = a
    c_02 = b
    h12, c12 = lstm_cell2(input, (h_02, c_02), attr)
    print(h12, c12)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def product(x):
    res = 1
    if isinstance(x, int):
        return x
    for i in range(len(x)):
        res *= x[i]

    return res


def _calculate_bounds(x):
    size = list(x.size())
    if len(size) > 2:
        x = x.view(size[0], product(*size[1]))
    sum1 = torch.sum(x[1:, :].abs(), dim=0)

    # print(sum1)
    lb = x[0] - sum1
    ub = x[0] + sum1
    x = x.view(size)
    return lb, ub


class AbstractLinear(nn.Module):

    def __init__(self, weight, bias):
        super(AbstractLinear, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        # print('linear input size {}'.format(x.size()))
        batch = x.size()[0]
        outs = []
        outs += [F.linear(x[0, :], self.weight, self.bias)]

        for i in range(1, batch):
            outs += [F.linear(x[i, :], self.weight)]

        out = torch.stack(outs, dim=0)

        # print('linear output size ', out.size())
        return out


class AbstractConv(nn.Module):

    def __init__(self, weight, bias, kernel_size, stride, padding):
        super(AbstractConv, self).__init__()
        self.weight = weight
        self.bias = bias
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        size = list(x.size())
        batch = x.size()[0]
        # print('cnn input size {}'.format(x.size()))
        outs = []
        outs += [F.conv2d(x[0].view(1, *size[1:]), self.weight, bias=self.bias, stride=self.stride,
                          padding=self.padding)]

        for i in range(1, batch):
            outs += [F.conv2d(x[i].view(1, *size[1:]), self.weight, bias=self.bias, stride=self.stride,
                              padding=self.padding)]

        out = torch.stack(outs, dim=0)

        # print('cnn output size ', out.size())

        return out


class AbstractRelu(nn.Module):

    def __init__(self, alpha):
        super(AbstractRelu, self).__init__()

    def forward(self, x):
        size = list(x.size())
        # print('relu input size {}'.format(x.size()))
        x = x.view(size[0], product(size[1:]))

        error_term, neuron = x.size()
        lb, ub = _calculate_bounds(x)

        # new_term = None

        z = (lb < 0).nonzero().flatten()
        upper_bound_less_than0 = (ub <= 0).nonzero().flatten()
        lower_bound_greater_than0 = (lb > 0).nonzero().flatten()
        lower_bound_less_than0 = (lb <= 0).nonzero().flatten()
        upper_bound_greater_than0 = (ub >= 0).nonzero().flatten()
        crossing_ = torch.from_numpy(np.intersect1d(lower_bound_less_than0, upper_bound_greater_than0))
        print(crossing_.size())
        new_error_terms = torch.zeros(product(list(crossing_.size())), neuron, dtype=torch.float32)
        # print(new_error_terms.size())
        alpha = ub[crossing_] / (ub[crossing_] - lb[crossing_])
        x[0, crossing_] = (alpha * x[0, crossing_]) - ((alpha * lb[crossing_]) / 2)
        x[1:, crossing_] = alpha * x[1:, crossing_]
        new_error_terms[crossing_, crossing_] = -1 * ((alpha * lb[crossing_]) / 2)
        x[:, upper_bound_less_than0] = 0

        print('x', x.size())
        print('new', new_error_terms.size())

        if crossing_.size()[0] == 0:
            x = x.view(*size)
            # print('relu output size {}'.format(x.size()))
            return x
        # new_error_terms = new_error_terms.view(list(new_error_terms.size())[0], x.size()[1:])
        out = torch.cat((x, new_error_terms), dim=0)

        # for n in range(neuron):
        #     l = lb[n].item()  # lower bound for nth relu node
        #     u = ub[n].item()  # upper bound for nth relu node
        #
        #     # print('lower bound {}, upper bound {}, neuron {}'.format(l, u, n))
        #     assert (l <= u)
        #
        #     if l == u:
        #         continue
        #
        #     if l > 0:
        #         pass  # y = x
        #     elif u <= 0:
        #         x[:, n] = 0  # y=0
        #     else:
        #         self.alpha = nn.Parameter(torch.tensor(u / (u - l)))
        #         assert (0 <= self.alpha <= 1)
        #         new_error_term = torch.zeros(1, neuron, dtype=torch.float32)
        #         # print(x[:, n])
        #         x[0, n] = self.alpha * x[0, n] - ((self.alpha * l) / 2)
        #         x[1:, n] = self.alpha * x[1:, n]
        #         new_error_term[0, n] = -1 * ((self.alpha * l) / 2)
        #         if new_term is None:
        #             new_term = new_error_term
        #         else:
        #             new_term = torch.cat((new_term, new_error_term))

        x = x.view(*size)

        # if new_term is None:
        #     # print('relu output size {}'.format(x.size()))
        #     return x

        # new_term = new_term.view(new_term.size()[0], *size[1:])
        # out = torch.cat((x, new_term), dim=0)
        # print('relu output size {}'.format(out.size()))
        return out

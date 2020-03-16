import argparse
import torch
import torch.nn as nn
from networks import FullyConnected, Conv
from abstract import AbstractRelu, AbstractLinear, AbstractConv

DEVICE = 'cpu'
INPUT_SIZE = 28


def product(x):
    res = 1

    for i in range(len(x)):
        res *= x[i]

    return res


def get_perturbed_image(image, eps):
    LB_N0 = image - eps
    UB_N0 = image + eps

    LB_N0 = torch.clamp(LB_N0, min=0, max=1)
    UB_N0 = torch.clamp(UB_N0, min=0, max=1)

    return LB_N0, UB_N0


def transform_input_to_zonotope(LB_N0, UB_N0):
    n_pixels = product(list(LB_N0.size()))

    errors = torch.zeros([n_pixels, 1, 28, 28], dtype=torch.float32)

    zonotope_center = (LB_N0 + UB_N0) / 2
    zonotope_dev = (UB_N0 - LB_N0) / 2

    zc = zonotope_center.view(1, n_pixels)
    zd = zonotope_dev.view(1, n_pixels)
    errors = errors.view(n_pixels, n_pixels)
    for i in range(n_pixels):
        errors[i, i] = zd[0, i]

    errors = errors.view(n_pixels, 1, 28, 28)
    return torch.cat((zonotope_center, errors), dim=0)


def verify(output, true_label, eps):
    nerrors, nclass = output.size()

    bounds = []
    for n in range(nclass):
        errors = output[:, n]
        sum1 = 0

        for e in range(1, nerrors):
            sum1 += abs(errors[e].item())

        l = errors[0].item() - sum1
        u = errors[0].item() + sum1

        # assert (l <= u)
        bounds += [(l, u)]
    # print(bounds)
    for i in range(len(bounds)):
        if i == true_label:
            continue

        if bounds[i][1] <= bounds[true_label][0]:  # upper bound of other labels should be less than lower bound of
            # true label
            continue
        else:  # if not we cannot verify the network since output bounds are overlapping
            return 0
    return 1


def analyze(net, inputs, eps, true_label):
    inputs = net.layers[0](inputs)
    LB_N0, UB_N0 = get_perturbed_image(inputs, eps)
    input_zonotope = transform_input_to_zonotope(LB_N0, UB_N0)
    layers = net.layers
    # print(layers)
    forward = input_zonotope

    # tmp = input_zonotope.view(785, 784)
    #
    # for i in range(1, 785):
    #     print('-----------------------')
    #     print('error term ', i)
    #     print(tmp[i, :])
    #     print('-----------------------')

    for layer_n, layer in enumerate(layers[1:]):
        # print('layer {} size is {}'.format(layer_n, forward.size()))
        if type(layer) == nn.Linear:
            forward = AbstractLinear(layer.weight, layer.bias)(forward)
        elif type(layer) == nn.Conv2d:
            forward = AbstractConv(layer.weight, layer.bias, None, layer.stride, layer.padding)(forward)
        elif type(layer) == nn.ReLU:
            forward = AbstractRelu(0.4)(forward)
        else:
            forward = layer(forward)
    return verify(forward, true_label, eps)


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
                        required=True,
                        help='Neural network to verify.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [150, 10], 10).to(DEVICE)
    elif args.net == 'conv4':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    elif args.net == 'conv5':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label
    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()

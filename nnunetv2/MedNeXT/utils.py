import torch
from torch import nn

def features_per_stage(start_num_features, num_stages, max_num_features):
    return tuple([min(max_num_features, start_num_features * 2 ** i) for i in range(num_stages)])

def _maybe_convert_scalar_to_list(scalar, n: int) -> list:
    if isinstance(scalar, int):
        scalar = [scalar] * n
    assert len(scalar) == n
    return list(scalar)

def maybe_convert_scalar_to_list(scalar, dim_or_convop) -> list:
    if not isinstance(dim_or_convop, int):
        dim_or_convop = convert_conv_op_to_dim(dim_or_convop)
    
    return _maybe_convert_scalar_to_list(scalar, dim_or_convop)

def convert_conv_op_to_dim(conv_op) -> int:

    if any(conv_op is c for c in [nn.Conv2d, nn.ConvTranspose2d]):
        return 2
    elif any(conv_op is c for c in [nn.Conv3d, nn.ConvTranspose3d]):
        return 3
    
    raise ValueError('Wtf? Only 3d and 2d convolutions are suppported. Got %s' % str(conv_op))

def _convert_conv_op_to_dim(conv_op) -> int:
    dim = conv_op.__name__[-2]
    return int(dim)

def convert_dim_to_conv_op(dim: int):
    conv_op = getattr(nn, f'Conv{dim}d', None)
    return conv_op

def get_matching_pool_op(dim_or_convop,
                         adaptive: bool = False,
                         pool_type: str = 'avg'):
    """
    :param dimension:
    :param adaptive:
    :param pool_type: either 'avg' or 'max'
    :return:
    """
    if not isinstance(dim_or_convop, int):
        dim_or_convop = convert_conv_op_to_dim(dim_or_convop)

    if dim_or_convop == 1:
        if pool_type == 'avg':
            if adaptive:
                return nn.AdaptiveAvgPool1d
            else:
                return nn.AvgPool1d
        elif pool_type == 'max':
            if adaptive:
                return nn.AdaptiveMaxPool1d
            else:
                return nn.MaxPool1d
    elif dim_or_convop == 2:
        if pool_type == 'avg':
            if adaptive:
                return nn.AdaptiveAvgPool2d
            else:
                return nn.AvgPool2d
        elif pool_type == 'max':
            if adaptive:
                return nn.AdaptiveMaxPool2d
            else:
                return nn.MaxPool2d
    elif dim_or_convop == 3:
        if pool_type == 'avg':
            if adaptive:
                return nn.AdaptiveAvgPool3d
            else:
                return nn.AvgPool3d
        elif pool_type == 'max':
            if adaptive:
                return nn.AdaptiveMaxPool3d
            else:
                return nn.MaxPool3d       

def get_matching_convtransp(dim_or_conv_op):
    """
    """
    if not isinstance(dim_or_conv_op, int):
        dim_or_conv_op = convert_conv_op_to_dim(dim_or_conv_op)

    if dim_or_conv_op == 1:
        return nn.ConvTranspose1d
    elif dim_or_conv_op == 2:
        return nn.ConvTranspose2d
    elif dim_or_conv_op == 3:
        return nn.ConvTranspose3d

def check_two_nets_are_equal(net1, net2, *, kwargs1: dict = None, kwargs2: dict = None, seed: int = 9999):
    
    torch.manual_seed(seed)
    if kwargs1 is not None:
        net1 = net1(**kwargs1)
        kwargs2 = kwargs2 or kwargs1

    torch.manual_seed(seed)
    if kwargs2 is not None:
        net2 = net2(**kwargs2)

    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

if __name__ == "__main__":
    net1 = nn.Conv3d
    kwargs = dict(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1, stride = 2)
    net2 = nn.Conv3d
    assert check_two_nets_are_equal(net1, net2, kwargs1=kwargs)

    assert convert_conv_op_to_dim(net1) == 3

    assert maybe_convert_scalar_to_list(3, dim_or_convop=3) == maybe_convert_scalar_to_list(3, nn.Conv3d) == [3, 3, 3]

    assert get_matching_pool_op(dim_or_convop = 3, adaptive=False, pool_type='avg') is \
        get_matching_pool_op(dim_or_convop=nn.Conv3d, adaptive=False, pool_type='avg') is \
        nn.AvgPool3d
    
    assert convert_dim_to_conv_op(3) is nn.Conv3d

    assert _convert_conv_op_to_dim(nn.Conv3d) == convert_conv_op_to_dim(nn.Conv3d) and \
        _convert_conv_op_to_dim(nn.Conv2d) == convert_conv_op_to_dim(nn.Conv2d)
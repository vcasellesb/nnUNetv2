import torch
from torch import nn
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks, ConvDropoutNormReLU

def test_ConvDroputNormReLU(input_size):
    conv = ConvDropoutNormReLU(nn.Conv2d, input_channels=1, output_channels=32, kernel_size=3, stride=1)
    print(conv.compute_conv_feature_map_size(input_size))

if __name__ == "__main__":
    # data = torch.rand((1, 3, 40, 32))

    # stx = StackedConvBlocks(2, nn.Conv2d, 24, 16, (3, 3), 2,
    #                         norm_op=nn.BatchNorm2d, nonlin=nn.ReLU, nonlin_kwargs={'inplace': True},
    #                         )
    # model = nn.Sequential(ConvDropoutNormReLU(nn.Conv2d,
    #                                           3, 24, 3, 1, True, nn.BatchNorm2d, {}, None, None, nn.LeakyReLU,
    #                                           {'inplace': True}),
    #                       stx)

    # print(model(data))
    # stx.compute_conv_feature_map_size((40, 32))

    test_ConvDroputNormReLU((128, 128))
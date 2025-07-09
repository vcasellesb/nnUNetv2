import torch
from torch import nn

from nnunetv2.MedNeXT.utils import convert_conv_op_to_dim, maybe_convert_scalar_to_list
from .encoder import MedNeXTEncoder
from .decoder import MedNeXTDecoder

class MedNeXT(nn.Module):
    def __init__(
            self,
            input_channels: int,
            num_stages: int,
            features_per_stage,
            expansion_ratio_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            conv_bias: bool,
            n_conv_per_stage,
            num_classes: int,
            n_conv_per_stage_decoder,
            norm_op = None,
            norm_op_kwargs = None,
            nonlin = None,
            nonlin_kwargs = None,
            deep_supervision: bool = True
    ):
        
        super().__init__()

        n_conv_per_stage = maybe_convert_scalar_to_list(n_conv_per_stage, num_stages)
        n_conv_per_stage_decoder = maybe_convert_scalar_to_list(n_conv_per_stage_decoder, num_stages - 1)
        expansion_ratio_per_stage = maybe_convert_scalar_to_list(expansion_ratio_per_stage, 2 * num_stages - 1)
        assert len(n_conv_per_stage) == num_stages, (
            "n_conv_per_stage must have as many entries as we have "
            f"resolution stages. here: {num_stages}. "
            f"n_conv_per_stage: {n_conv_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (num_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {num_stages} "
            f"stages, so it should have {num_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        
        self.encoder = MedNeXTEncoder(
            input_channels,
            num_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            expansion_ratio_per_stage=expansion_ratio_per_stage[:num_stages],
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            conv_bias=conv_bias,
            return_skips=True
        )

        self.decoder = MedNeXTDecoder(
            self.encoder, num_classes, n_conv_per_stage_decoder,
            expansion_ratio_per_stage = expansion_ratio_per_stage[num_stages:],
            deep_supervision=deep_supervision, conv_bias=conv_bias
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )
from copy import deepcopy
import numpy as np

from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props

def with_workaround(
        patch_size: tuple | np.ndarray, 
        spacing: tuple | np.ndarray, 
        shape_must_be_divisible_by: tuple | np.ndarray, 
        axis_to_be_reduced: int
):
    
    UNet_featuremap_min_edge_length = 4

    num_pools, strides_per_stage, kernel_sizes_per_tage, patch_size, shape_must_be_divisible_by  = \
        get_pool_and_conv_props(spacing, patch_size, UNet_featuremap_min_edge_length, 99999)
    
    assert list(patch_size) == [256, 256, 256] and num_pools == [6, 6, 6], num_pools

    patch_size = list(patch_size)
    tmp = deepcopy(patch_size)
    tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
    assert tmp == [256, 256, 192]
    num_pools, _, _, _, shape_must_be_divisible_by = \
        get_pool_and_conv_props(spacing, tmp,
                                UNet_featuremap_min_edge_length,
                                999999)
    patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
    assert num_pools == [6, 6, 5]
    assert patch_size == [256, 256, 224]
    # now recompute topology
    network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
    shape_must_be_divisible_by = get_pool_and_conv_props(spacing, patch_size,
                                                            UNet_featuremap_min_edge_length,
                                                            999999)
    assert list(patch_size) == [256, 256, 224] and network_num_pool_per_axis == [6, 6, 5]
    return (network_num_pool_per_axis,
            pool_op_kernel_sizes, 
            conv_kernel_sizes, 
            patch_size, 
            shape_must_be_divisible_by)

if __name__ == "__main__":
    res = with_workaround(
        patch_size=(256, 256, 256),
        spacing = (1.,1.,1.),
        shape_must_be_divisible_by=(2**5, ) * 3,
        axis_to_be_reduced=-1
    )
    print(res)
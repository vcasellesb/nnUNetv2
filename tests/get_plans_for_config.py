from copy import deepcopy
import numpy as np

from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props

# I wanna test lines 321 - 345 from nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py
# Essentially what this does is update the patch size to fit the memory budget. I just wanna simulate what changes
# when get_pool_and_conv_props is called multiple times

def without_workaround(
        patch_size: tuple | np.ndarray, 
        spacing: tuple | np.ndarray, 
        shape_must_be_divisible_by: tuple | np.ndarray, 
        axis_to_be_reduced: int
    ):
    
    UNet_featuremap_min_edge_length = 4
    
    patch_size = list(patch_size)
    # tmp = deepcopy(patch_size)
    # tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
    # _, _, _, _, shape_must_be_divisible_by = \
    #     get_pool_and_conv_props(spacing, tmp,
    #                             UNet_featuremap_min_edge_length,
    #                             999999)
    patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

    # now recompute topology
    network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
    shape_must_be_divisible_by = get_pool_and_conv_props(spacing, patch_size,
                                                            UNet_featuremap_min_edge_length,
                                                            999999)
    return (network_num_pool_per_axis, 
            pool_op_kernel_sizes, 
            conv_kernel_sizes, 
            patch_size, 
            shape_must_be_divisible_by)

def with_workaround(
        patch_size: tuple | np.ndarray, 
        spacing: tuple | np.ndarray, 
        shape_must_be_divisible_by: tuple | np.ndarray, 
        axis_to_be_reduced: int
):
    
    UNet_featuremap_min_edge_length = 4
    
    patch_size = list(patch_size)
    tmp = deepcopy(patch_size)
    tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
    _, _, _, _, shape_must_be_divisible_by = \
        get_pool_and_conv_props(spacing, tmp,
                                UNet_featuremap_min_edge_length,
                                999999)
    patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

    # now recompute topology
    network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
    shape_must_be_divisible_by = get_pool_and_conv_props(spacing, patch_size,
                                                            UNet_featuremap_min_edge_length,
                                                            999999)
    return (network_num_pool_per_axis, 
            pool_op_kernel_sizes, 
            conv_kernel_sizes, 
            patch_size, 
            shape_must_be_divisible_by)

def main(
        patch_size,
        spacing,
        shape_must_be_divisible_by,
        axis_to_be_reduced
):
    og_ps = deepcopy(patch_size)
    network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, \
        this_patch_size, shape_must_be_divisible_by = without_workaround(patch_size, spacing, shape_must_be_divisible_by, axis_to_be_reduced)
    
    print('Results without the tmp workaround: ')
    print(f'{network_num_pool_per_axis = }')
    print(f'{this_patch_size = }')
    print(f'{shape_must_be_divisible_by = }')
    
    assert patch_size == og_ps

    network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, \
        this_patch_size, shape_must_be_divisible_by = with_workaround(patch_size, spacing, shape_must_be_divisible_by, axis_to_be_reduced)
    
    print()
    print('Results with the tmp workaround: ')
    print(f'{network_num_pool_per_axis = }')
    print(f'{this_patch_size = }')
    print(f'{shape_must_be_divisible_by = }')

# Insight gathered:
# What is going on here, is that if we reduce the patch size so that the computational requirements are lower,
# we effectively affect the number of pooling operations that we do (downsampling). For example assume we were 
# performing m ops before. Then, potentially, the patch_size does not have to be divisible by 2 ** m anymore, 
# but by 2 ** (m-1). Hence, we can increase the patch size considerably - we don't have to substract 2**m anymore, 
# rather 2**(m-1), allowing us to increasing the input size.
# TODO: Could this re-run of get_pool_and_conv_props be actually avoided and just substract 2 ** (m-1) from patch_size
# instead of rerunning the whole thing?

if __name__ == "__main__":
    patch_size = (256, ) * 3
    spacing = (1., ) * 3
    shape_must_be_divisible_by = (64, ) * 3
    axis_to_be_reduced = 2
    main(
        patch_size,
        spacing,
        shape_must_be_divisible_by,
        axis_to_be_reduced
    )
import numpy as np


def pad_shape(shape, must_be_divisible_by):
    """
    pads shape so that it is divisible by must_be_divisible_by
    :param shape:
    :param must_be_divisible_by:
    :return:
    """
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)

    new_shp = [shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i] for i in range(len(shape))]

    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shp[i] -= must_be_divisible_by[i]
    new_shp = np.array(new_shp).astype(int)
    return new_shp

def pad_shape2(shape, must_be_divisible_by):
    
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)
    
    new_shape = [shape[i] + bool(tmp:=shape[i] % must_be_divisible_by[i]) * must_be_divisible_by[i] - tmp for i in range(len(shape))]
    new_shape = np.array(new_shape).astype(int)
    return new_shape

# yet another way
def pad_shape3(shape, must_be_divisible_by):

    dim = len(shape)
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * dim
    else:
        assert len(must_be_divisible_by) == dim

    remainders = [shape[i] % must_be_divisible_by[i] for i in range(dim)]

    new_shp = [shape[i] + (must_be_divisible_by[i] - remainders[i]) * bool(remainders[i]) for i in range(dim)]

    return np.array(new_shp).astype(int)

if __name__ == "__main__":
    shape = (120, ) * 3
    must_be_divisible_by = 25
    assert (pad_shape(shape, must_be_divisible_by) == pad_shape2(shape, must_be_divisible_by)).all(), pad_shape2(shape, must_be_divisible_by)
    assert (pad_shape(shape, must_be_divisible_by) == np.array([125] * 3)).all()

    shape = (220, 220, 220)
    must_be_divisible_by = 50
    assert (pad_shape(shape, must_be_divisible_by) == pad_shape2(shape, must_be_divisible_by)).all(), pad_shape2(shape, must_be_divisible_by)
    assert (pad_shape(shape, must_be_divisible_by) == np.array([250] * 3)).all()

    shape, must_be_div_by = (119, 119, 249), (20, 20, 20)
    assert (pad_shape(shape, must_be_div_by) == pad_shape3(shape, must_be_div_by)).all()

    shape, must_be_div_by = (90, 90, 90), (16, 16, 16)
    assert (pad_shape(shape, must_be_div_by) == pad_shape3(shape, must_be_div_by)).all() and (pad_shape3(shape, must_be_div_by) == np.array([96] * 3)).all()
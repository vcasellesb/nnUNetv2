from typing import Iterable
import numpy as np

def determine_transpose(target_spacing):
    max_spacing_axis = np.argmax(target_spacing)
    remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
    transpose_forward = [max_spacing_axis] + remaining_axes
    transpose_backward = [np.argwhere(np.array(transpose_forward) == i)[0][0] for i in range(3)]
    return transpose_forward, transpose_backward

def determine_transpose2(target_spacing: Iterable[float]):
    max_spacing = max(target_spacing)
    max_spacing_axis = target_spacing.index(max_spacing)
    transpose_forward = [max_spacing_axis] + [i for i in range(3) if i != max_spacing_axis]
    transpose_backward = [transpose_forward.index(i) for i in range(3)]
    return transpose_forward, transpose_backward

def test(spacing):
    results1 = determine_transpose(spacing)
    results2 = determine_transpose2(spacing)
    assert all(i == j for i, j in zip(results1, results2))

if __name__ == "__main__":
    test((1., 1., 5.))
import os
import shutil
import nibabel as nib
import numpy as np
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero

def load_file(nii_path: str) -> nib.Nifti1Image:
    return nib.load(nii_path)

def is_anisotropic(spacing, shape):
    biggest_spacing = np.argmax(spacing)
    second_biggest_spacing = max((spacing[i] for i in range(len(spacing)) if i != biggest_spacing))
    shape_biggest_spacing = shape[biggest_spacing]
    shape_second_biggest_spacing = min((shape[i] for i in range(len(shape)) if i != biggest_spacing))
    
    aniso1 = spacing[biggest_spacing] > 3 * second_biggest_spacing
    aniso2 = shape_biggest_spacing * 3 < shape_second_biggest_spacing
    
    if aniso1 and aniso2:
        return biggest_spacing

def get_shape_after_crop(data: np.ndarray, seg):
    data, *_ = crop_to_nonzero(data, seg)
    return data.shape[1:]

def main(list_of_datas: list[str], list_of_segs: list[str]):
    spacings = []
    shapes = []
    shapes_after_cropping = []
    for data, seg in zip(list_of_datas, list_of_segs):
        nii = load_file(data)
        seg = load_file(seg)
        spacing = list(np.abs(nii.header.get_zooms()))
        shape = nii.header.get_data_shape()
        spacings.append(spacing)
        shapes.append(shape)
        shapes_after_cropping.append(get_shape_after_crop(np.asanyarray(nii.dataobj)[None], 
                                                          np.asanyarray(seg.dataobj)[None].astype(np.int8)))

    spacings = np.vstack(spacings)
    shapes_after_cropping = np.vstack(shapes_after_cropping)
    shapes = np.vstack(shapes)
    target = np.percentile(spacings, 50, 0)
    target_shape = np.percentile(shapes_after_cropping, 50, 0)
    maybe_target_shape = np.percentile(shapes, 50, 0)
    # the following line tries to check that the results reported in nnUNet's paper are right
    # https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf
    # They must have changed something in the algo, because median image shape is not [256, 237, 9] anymore (see figure SN3.1)
    # I have tried on pre and post cropping shapes, results are the same
    # assert (target_shape == np.array([256, 237, 9])).all(), target_shape

    is_aniso = is_anisotropic(target, target_shape)
    if is_aniso:
        target[is_aniso] = np.percentile(spacings, 10, 0)[is_aniso]
    return target

join = os.path.join
def use_nnunet(datas, segs):
    nnUNet_raw = 'nnUNet_raw/Dataset_002ACDC'
    imagesTr = join(nnUNet_raw, 'imagesTr')
    labelsTr = join(nnUNet_raw, 'labelsTr')
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    idx = 1
    for d, s in zip(datas, segs):
        shutil.copy(d, join(imagesTr, f'ACDC{idx}_0000.nii.gz'))
        shutil.copy(d, join(labelsTr, f'ACDC{idx}.nii.gz'))
        idx+=1
    
    command = 'nnUNet_raw=nnUNet_raw nnUNet_preprocessed=nnUNet_preprocessed nnUNet_results="" \
               nnUNetv2_plan_and_preprocess -d 002 -np 14 --no_pp -npfp 14'
    os.system(command)

if __name__ == "__main__":
    from glob import glob
    files = glob('database/training/patient*/*frame*.nii')
    segs = [i for i in files if i.endswith('_gt.nii')]
    datas = list(set(files) - set(segs))
    datas.sort()
    datas.sort(key=len)
    segs.sort()
    segs.sort(key=len)
    assert len(datas) == len(segs) == 200
    use_nnunet(datas, segs)
    # target_spacing = main(datas, segs)
    # assert (target_spacing == np.array([5, 1.5625, 1.5625][::-1])).all(), target_spacing
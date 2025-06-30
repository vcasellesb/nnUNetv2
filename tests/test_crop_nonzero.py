import nibabel as nib
import numpy as np

from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
import config

def main(data_path: str, seg_path: str):
    im: nib.Nifti1Image = nib.load(data_path)
    data = np.asanyarray(im.dataobj)
    print('input dim:', data.shape)
    seg = np.asanyarray(nib.load(seg_path).dataobj)

    data, seg, bbox = crop_to_nonzero(data[None], seg[None])
    print('Output dim:', data.squeeze(0).shape)
    nib.save(
        nib.Nifti1Image(data.squeeze(0), affine=im.affine),
        'test_data.nii.gz'
    )
    nib.save(
        nib.Nifti1Image(seg.squeeze(0), affine=im.affine),
        'test_seg.nii.gz'
    )

if __name__ == "__main__":
    import sys
    main(
        config.IMAGE_FILE,
        config.SEG_FILE
    )
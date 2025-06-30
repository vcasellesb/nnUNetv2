import nibabel as nib
import numpy as np

import tests.config as config

def main(im_path, seg_path):
    data = np.asanyarray(nib.load(im_path).dataobj)
    assert len(data) == data.shape[0]
    data = data[None]
    assert len(data) == 1


if __name__ == "__main__":
    main(
        config.IMAGE_FILE,
        config.SEG_FILE
    )

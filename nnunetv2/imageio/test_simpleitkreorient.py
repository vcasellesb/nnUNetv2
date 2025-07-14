import re
from os.path import join, exists
from os import listdir, system
import SimpleITK as sitk

from .simpleitk_reader_writer import SimpleITKIOReorientOnLoad
from ._reorient_utils import _get_orientation_from_direction

def _get_identifiers(folder: str, file_ending: str | tuple[str, ...]) -> set[str]:
    data_regex = re.compile(r'([IM][fb])?_(\d{4})' + file_ending)
    files = [f for f in listdir(folder) if f.endswith(file_ending)]
    return set(
        map(
            lambda x: x.groups()[-1],
            filter(None, map(
                data_regex.search,
                files
                )
            )
        )
    )

def get_identifiers(folder: str, file_ending: str | tuple[str, ...], sort: bool = True) -> list[str]:
    identifiers = list(_get_identifiers(folder, file_ending))
    if sort:
        identifiers.sort()
    return identifiers

def _generate_iterable_with_fnames(folder: str, identifier: str, file_ending: str) -> tuple[tuple[str, str], str]:
    image_fnames = ('If_' + identifier + file_ending, 'Mb_' + identifier + file_ending)
    image_fnames = tuple(map(lambda x: join(folder, x), image_fnames))
    seg_fname = join(folder, 'Mf_' + identifier + file_ending)
    assert all(map(exists, image_fnames + (seg_fname, ))), str(image_fnames) + '\n' + seg_fname
    return {'images': image_fnames, 'seg': seg_fname}

def generate_iterable_with_fnames(folder: str, file_ending: str = '.nii.gz') -> dict[str, dict]:
    return {ii: _generate_iterable_with_fnames(folder, ii, file_ending) for ii in get_identifiers(folder, file_ending)}

def main(folder: str):
    dataset = generate_iterable_with_fnames(folder)
    rw = SimpleITKIOReorientOnLoad()
    for case in dataset:
        seg, properties = rw.read_seg(dataset[case]['seg'])
        original_orient = properties['sitk_stuff']['original_orient']

        rw.write_seg(seg[0], 'test_seg.nii.gz', properties)

        orient = _get_orientation_from_direction(sitk.ReadImage('test_seg.nii.gz').GetDirection())
        assert original_orient == orient
        system('rm test_seg.nii.gz')

if __name__ == "__main__":
    main('../timelessegv2/test-out')
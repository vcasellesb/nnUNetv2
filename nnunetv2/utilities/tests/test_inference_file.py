from multiprocessing import Pool
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import re

from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder

def _get_identifiers(folder: str) -> set[str]:
    data_regex = re.compile(LISA_REGEX)
    return set(
        map(
            lambda x: x.groups()[-1],
            filter(None, map(
                data_regex.search,
                listdir(folder)
                )
            )
        )
    )

def get_identifiers(folder: str, sort: bool = True) -> list[str]:
    identifiers = list(_get_identifiers(folder))
    if sort:
        identifiers.sort()
    return identifiers



if __name__ == "__main__":
    print(create_lists_from_splitted_dataset_folder('nnUNet_raw/Dataset003_LISA/imagesTr', '.nii.gz'))
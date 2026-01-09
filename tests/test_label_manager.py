from nnunetv2.utilities.label_handling.label_handling import LabelManager
from batchgenerators.utilities.file_and_folder_operations import load_json


def main(dataset_json):
    labels_dict = load_json(dataset_json)['labels']
    label_manager = LabelManager(
        labels_dict,
        None
    )
    print(label_manager.all_labels)

if __name__ == "__main__":
    main(
        'nnUNet_preprocessed/Dataset003_mine/dataset.json'
    )
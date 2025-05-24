from typing import Union, List, Tuple

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import nnUNetPlannerResEncM

# We decide to lower the minimum edge length of feature maps at the lowest resolution
# By default, nnUNet uses 4. However, that might be too small for MS lesions.
# Some scouting of the state of the art:
#   * LST-AI uses 12
#   * HD-MS uses 4
#   * La Rosa uses 20ish
#   

class ExperimentPlannerLessStages(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetLessStagesPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False, 
                 UNet_featuremap_min_edge_length: int = 8):
                
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, 
                         preprocessor_name, plans_name, overwrite_target_spacing, 
                         suppress_transpose)
        
        print(f'Using a custom experiment planner, with name "{self.__class__.__name__}", which has a '
                  f'custom minimum number of feature map edge length (at the lowest resolution). '
                  f'It has been set to: {UNet_featuremap_min_edge_length}.')
        
        self.UNet_featuremap_min_edge_length = UNet_featuremap_min_edge_length


class nnUNetPlannerResEncMLessStages(nnUNetPlannerResEncM):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetMLessStagesPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False,
                 UNet_featuremap_min_edge_length: int = 8):
        
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb,
                         preprocessor_name, plans_name, overwrite_target_spacing,
                         suppress_transpose)
        
        print(f'Using a custom experiment planner, with name "{nnUNetPlannerResEncMLessStages.__name__}", which has a '
              f'custom minimum number of feature map edge length (at the lowest resolution). '
              f'It has been set to: {UNet_featuremap_min_edge_length}.')
        
        self.UNet_featuremap_min_edge_length = UNet_featuremap_min_edge_length
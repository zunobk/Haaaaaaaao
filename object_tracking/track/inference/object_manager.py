from typing import Union, List, Dict
import torch
from object_tracking.track.inference.object_info import ObjectInfo

class ObjectManager:
    """
    Object IDs are immutable. The same ID always represent the same object.
    Temporary IDs are the positions of each object in the tensor. It changes as objects get removed.
    Temporary IDs start from 1.
    """
    def __init__(self):
        self.obj_to_tmp_id: Dict[ObjectInfo, int] = {}
        self.tmp_id_to_obj: Dict[int, ObjectInfo] = {}
        self.obj_id_to_obj: Dict[int, ObjectInfo] = {}

        self.all_historical_object_ids: List[int] = []
    
    @property
    def all_obj_ids(self) -> List[int]:
        return [k.id for k in self.obj_to_tmp_id]

    @property
    def num_obj(self) -> int:
        return len(self.obj_to_tmp_id)
    
    def has_all(self, objects: List[int]) -> bool:
        for obj in objects:
            if obj not in self.obj_to_tmp_id:
                return False
        return True
    
    def add_new_objects(
            self, objects: Union[List[ObjectInfo], ObjectInfo,
                                 List[int]]) -> (List[int], List[int]):
        if not isinstance(objects, list):
            objects = [objects]

        corresponding_tmp_ids = []
        corresponding_obj_ids = []
        for obj in objects:
            if isinstance(obj, int):
                obj = ObjectInfo(id=obj)

            if obj in self.obj_to_tmp_id:
                # old object
                corresponding_tmp_ids.append(self.obj_to_tmp_id[obj])
                corresponding_obj_ids.append(obj.id)
            else:
                # new object
                new_obj = ObjectInfo(id=obj)

                # new object
                new_tmp_id = len(self.obj_to_tmp_id) + 1
                self.obj_to_tmp_id[new_obj] = new_tmp_id
                self.tmp_id_to_obj[new_tmp_id] = new_obj
                self.all_historical_object_ids.append(new_obj.id)
                corresponding_tmp_ids.append(new_tmp_id)
                corresponding_obj_ids.append(new_obj.id)

        self._recompute_obj_id_to_obj_mapping()
        assert corresponding_tmp_ids == sorted(corresponding_tmp_ids)
        return corresponding_tmp_ids, corresponding_obj_ids
    
    def _recompute_obj_id_to_obj_mapping(self) -> None:
        self.obj_id_to_obj = {obj.id: obj for obj in self.obj_to_tmp_id}
        
    def realize_dict(self, obj_dict, dim=1) -> torch.Tensor:
        # turns a dict indexed by obj id into a tensor, ordered by tmp IDs
        output = []
        for _, obj in self.tmp_id_to_obj.items():
            if obj.id not in obj_dict:
                raise NotImplementedError
            output.append(obj_dict[obj.id])
        output = torch.stack(output, dim=dim)
        return output
    
    def find_tmp_by_id(self, obj_id) -> int:
        return self.obj_to_tmp_id[self.obj_id_to_obj[obj_id]]
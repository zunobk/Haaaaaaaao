class ObjectInfo:
    """
    Store meta information for an object
    """
    def __init__(self, id: int):
        self.id = id
        self.poke_count = 0  # count number of detections missed
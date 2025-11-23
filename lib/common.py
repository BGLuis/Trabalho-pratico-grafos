class Vertex:
    def __init__(self, label: str, weight: float):
        self._label = label
        self._weight = weight

    def get_label(self) -> str:
        return self._label

    def get_weight(self) -> float:
        return self._weight

    def set_label(self, new_name: str):
        self._label = new_name

    def set_weight(self, new_weight: float):
        if new_weight < 0:
            raise ValueError("Vertex weight cannot be negative.")
        self._weight = new_weight

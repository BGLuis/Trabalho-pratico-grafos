from vertex import Vertex


class Edge:
    def __init__(self, source: Vertex, target: Vertex):
        self._weight = 0
        self._source = source
        self._target = target

    def get_weight(self) -> float:
        return self._weight

    def get_source(self) -> Vertex:
        return self._source

    def get_target(self) -> Vertex:
        return self._target

    def set_weight(self, new_weight: float):
        if new_weight < 0:
            raise ValueError("O peso da aresta não pode ser negativo neste modelo.")
        self._weight = new_weight

    def set_source(self, new_source: int):
        if new_source < 0:
            raise ValueError("O índice da origem deve ser não-negativo.")
        self._source = new_source

    def set_target(self, new_target: int):
        if new_target < 0:
            raise ValueError("O índice do destino deve ser não-negativo.")
        self._target = new_target

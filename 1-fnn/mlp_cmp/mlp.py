from . import activation, loss, fc

class MLP:
    def __init__(self, in_size: int, out_size: int) -> None:
        self.in_size = in_size
        self.out_size = out_size
        self.layers = []
        pass

    def add_layer(self, layer: fc.FC) -> None:
        self.layers.append(layer)

    def train():
        pass

    def validate():
        pass

    def test():
        pass

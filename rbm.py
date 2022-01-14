import torch

class RBM:
    def __init__(self, n_visible: int, n_hidden: int, W = None, a = None, b = None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if W is None:
            self.W = torch.zeros(self.n_visible, self.n_hidden)
        else:
            assert len(W.shape) == 2 and W.shape[0] == self.n_visible and W.shape[1] == self.n_hidden
            self.W = W

        if a is None:
            self.a = torch.zeros(self.n_visible)
        else:
            assert len(a.shape) == 1 and a.shape[0] == n_visible
            self.a = a

        if b is None:
            self.b = torch.zeros(self.n_hidden)
        else:
            assert len(b.shape) == 1 and b.shape[0] == n_hidden
            self.b = b
    
    @staticmethod
    def from_weights(W, a, b):
        assert len(W.shape) == 2 and len(a.shape) == 1 and len(b.shape) == 1
        assert W.shape[0] == a.shape[0] and W.shape[1] == b.shape[0]

        n_visible = a.shape[0]
        n_hidden = b.shape[0]

        return RBM(n_visible, n_hidden, W, a, b)

    def to_hidden(self, visible):
        """get a new hidden layer from a given visible layer using sigmoid function """
        b = self.b
        vW = torch.matmul(visible, self.W)
        return torch.sigmoid(vW + b)

    def to_visible(self, hidden):
        """get a new visible layer from a given hidden layer using sigmoid function """
        a = self.a
        Wh = torch.matmul(hidden, self.W.t())
        return torch.sigmoid(Wh + a)


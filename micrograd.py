import math

class Value:

    def __init__(self, data, _children=()) -> None:
        self.data = data
        self._prev = set(_children)
        self._backward = lambda: None
        self.grad = 0.0

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other))
        

        def _backward():
            # we use += instead of = because it gets overriden if we use the variable again
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out
    



    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ))

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out
    
    def backward(self):
        self.grad = 1.0
        holding = [self]

        while holding:
            current = holding.pop()
            current._backward()
            holding.extend(current._prev)



    


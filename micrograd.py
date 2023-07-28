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



    


'''# inputs x1,x2
x1 = Value(2.0)
x2 = Value(9.0)
# weights w1,w2
w1 = Value(-3.0)
w2 = Value(1.0)
# bias of the neuron
b = Value(6.8813735870195432)
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; 
x2w2 = x2*w2; 
x1w1x2w2 = x1w1 + x2w2; 
n = x1w1x2w2 + b
o = n.tanh()
o.backward()
print(o.grad)'''

a = Value(3.0)
b = a + a
b.backward()
print(a.grad)
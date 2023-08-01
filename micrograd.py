import math

# Basic unit of operations
class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self._prev = set(_children)
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value({self.data})"
    
    # addition
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad +=  out.grad
            other.grad +=  out.grad
        
        self._backward = _backward
        return out
    

    # subtraction
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other)) 
        def _backward():
            self.grad +=  out.grad
            other.grad +=  out.grad
        
        self._backward = _backward      
        return out 


    # multiplication
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad +=  other.data * out.grad
            other.grad += self.data * out.grad
        
        self._backward = _backward
        return out
    

    #division
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        if other.data != 0:
            out = Value(self.data/ other.data, (self, other))
            return out
        else:
            raise ZeroDivisionError("Dividing by zero!")
        




    # exponent
    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other))

        def _backward():
            self.grad += (other.data * self.data ** (other.data -1)) * out.grad
        out._backward = _backward

        return out
    

    #tanh (activation function)
    def tanh(self):
        s = self.data
        t = (math.exp(2*s) -1)/(math.exp(2*s) + 1)
        out = Value(t, (self, )) 

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out
    

    #backprop with topological sort. See backprop.md for detailed example with an example
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                print("topo is", topo)
        
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()





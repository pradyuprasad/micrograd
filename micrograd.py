import math
import random
from typing import Any

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
            self.grad +=  +out.grad
            other.grad +=  -out.grad
        
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



#Foward pass of a MLP
class Neuron:
    # The Neuron class implements one matrix multiplication. Tanh(Weights times data + bias). 
    def __init__(self, nin): #nin is the size of the input
        self.w = [Value(random.uniform(-1, 1)) for i in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x) -> Any:
        if len(x) != len(self.w):
            raise ValueError("length of x is not the same as input to Neuron")
        else: 
            n = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
            out = n.tanh()
            return out


class Layer:
    def __init__(self, nin, nout): #nin is the length of x. nout is the number of neurons we want in this layer
        self.neurons = [Neuron(nin) for _ in range(nout)] #make a list of "nout" Neurons each having nin variables
    def __call__(self, x):
        #out put the forward pass of each neuron as a list
        out = [n(x) for n in self.neurons]
        return out

class MLP:
    def __init__(self, nin, nouts):#nin is the size of the input. nouts is a list of the *number* of neurons in each layer
        size = [nin] + nouts
        self.MLP = [Layer(size[i], size[i+1]) for i in range(len(size)-1)] # we make layers of each consecutive combination in size

    def __call__(self, x):
        for n in self.MLP:
            x = n(x) #we start with the input. then we apply each layer to x resulting in the answer 
        return x


x = [2.0, 3.0, -1.0, 90]
n = MLP(len(x), [4, 4, 1])
print(n(x))









# Incomplete, Works in Progress


# How does topological sort work?

Recall our function from micrograd:



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






Lets say we have 2 variables (x1, x2) and 2 weights (w1, w2).
~~~
w1x1 := x1 * w1 and w2x2 := x2 * w2

n := w1x1 + w2x2. 

o: tanh(n)
~~~
----


when we call o.backward()
topo = []
visited = set()

now we build_topo(o)

o is not in visited, o._prev is n
visited = (o)
----

now we do build_topo(n)

n is also not in visited
visited = (o, n)
n._prev = w1x1 and w2x2
----

for w1x1:
build_topo(w1x1)
visited = (o, n, w1x1)
w1x1._prev = w1 and x1
----

for w1
visited = (o, n, w1x1, w1)
w1._prev is empty
topo = [w1]
----

now we go to x1[^1]


for x1
visited = (o, n, w1x1, w1, x1)
x1._prev is empty
topo = [w1, x1]


We have looped over all children of w1x1. So, we add it to to topo
topo = [w1, x1, w1x1]
----

Recall that w1x1 was a child of n, and we are yet to loop over the children of w2x2. I'll save you the details and just give you the values of topo and visited

[^1]: Why do we go back to x1? Recall that build_topo(w1) is called *inside* build_topo(w1x1). So when this function terminates, we move to the next child in "for child in v._prev"



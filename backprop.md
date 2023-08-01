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





# defining the variables
Lets say we have 2 variables (x1, x2) and 2 weights (w1, and w2)

~~~
w1x1= w1*x1; w2x2 = w2*x2
n = w1x1 + w1x2
o = n.tanh()
~~~

# performing the operations

## Topological Sort via Depth First Search
When we call o.backward(), first we call build_topo(o)

topo = []
visited = ()

* As o is not in visited, visited = (o)

* The child of o is n. As n isn't in visited, visited = (o, n)

* The 2 child nodes of n are w1x1 and w2x2. I'll only show an example for w1x1 as w2x2 is the equivalent, just with different variables. 

* w1x1 isn't in visited, so visited = (o, n, w1x1). The two child variables are w1 and x1. 

* w1 isn't in visited, so visited = (o, n, w1x1, w1). As it has no children we can directly add it to topo. topo = [w1]. Same for x1: x1 isn't in visited, so visited = (o, n, w1x1, w1, x1). As it has no children we can directly add it to topo. topo = [w1, x1]

* We've now finished with w1x1 and all children nodes. We can now add w1x1 to topo as well. topo = [w1, x1, w1x1]. 

* Now, we repeat that for w2x2. I'll save you the details and give you the final values of visited and topo. visited = (o, n, w1x1, w1, x1, w2, x2, w2x2). topo = [w1, x1, w1x1, w2, x2, w2x2]

* Finally we can add n, and then o to topo.topo = [w1, x1, w1x1, w2, x2, w2x2, n, o]

After this, its quite clear. You reverse topo, and find the gradient of each element in topo in that order


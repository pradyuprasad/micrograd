# Incomplete, Works in Progress


How does topological sort work?

Lets say we have 2 variables (x1, x2) and 2 weights (w1, w2). n := x1w1 + x2w2. 

o: tanh(n)

when we call o.backward()
topo = []
visited = set()

now we build_topo(o)

o is not in visited, o._prev is n
visited = (o)

now we do build_topo(n)

n is also not in visited
visited = (o, n)
n._prev = w1x1 and w2x2

for w1x1:
build_topo(w1x1)
visited = (o, n, w1x1)
w1x1._prev = w1 and x1


for w1
visited = (o, n, w1x1, w1)
w1._prev is empty

now we go to x1[^1]

[^1]: testing



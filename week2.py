from graphviz import Digraph
import math
import random

#Value Class

class Value:
    def __init__(self, value, _children=(), _op='', label=''):
        self.value = value
        self._prev = _children
        self._op = _op
        self.label = label
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.value})"

    def __add__(self, other):
        return Value(self.value + other.value, (self, other), '+')

    def __mul__(self, other):
        return Value(self.value * other.value, (self, other), '*')

a = Value(2)
a.label = 'a'
b = Value(3)
b.label = 'b'

c = a + b
c.label = 'c'
d = Value(4)
d.label = 'd'

print(c.value)
print(c._prev)
print(c._op)

class Value:
    def __init__(self, value, _children=(), _op='', label=''):
        self.value = value
        self._prev = _children
        self._op = _op
        self.label = label
        self.grad = 0.0

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(
            self.value + other.value,
            (self, other),
            '+'
        )

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(
            self.value * other.value,
            (self, other),
            '*'
        )

    def __repr__(self):
        return f"Value(data={self.value})"

    def tanh(self):
        x = self.value
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        return out

    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(
            self.value / other.value,
            (self, other),
            '/'
        )

    def __pow__(self, other):
        return Value(
            self.value ** other,
            (self,),
            '**'
        )

    def exp(self):
        x = self.value
        out = Value(math.exp(x), (self,), 'exp')
        return out

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(
            self.value - other.value,
            (self, other),
            '-'
        )
    
    def backward(self):

        topology = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topology.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topology):
            if v._op == '+':
                for child in v._prev:
                    child.grad += 1.0 * v.grad
            elif v._op == '*':
                x, y = v._prev
                x.grad += y.value * v.grad
                y.grad += x.value * v.grad
            elif v._op == 'tanh':
                x, = v._prev
                x.grad += (1 - v.value ** 2) * v.grad

    def tanh2(self):
        x = self

        e = (x * 2).exp()

        numerator = e - 1
        denominator = e + 1

        return numerator / denominator

    def backward2(self):

        topology = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topology.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topology):
            if v._op == '+':
                for child in v._prev:
                    child.grad += 1.0 * v.grad
            elif v._op == '*':
                x, y = v._prev
                x.grad += y.value * v.grad
                y.grad += x.value * v.grad
            elif v._op == 'tanh':
                x, = v._prev
                x.grad += (1 - v.value ** 2) * v.grad
            elif v._op == '/':
                x, y = v._prev

                print("DIVISION")
                print("x =", x.value)
                print("y =", y.value)

                x.grad += (1 / y.value) * v.grad
                y.grad += (-x.value / (y.value ** 2)) * v.grad
            elif v._op == '**':
                x, = v._prev
                x.grad += (2 * x.value) * v.grad
            elif v._op == 'exp':
                x, = v._prev
                x.grad += (v.value) * v.grad
            elif v._op == '-':
                x, y = v._prev
                x.grad += 1.0 * v.grad
                y.grad += -1.0 * v.grad
            

def trace(root):
    nodes = set()
    edges = set()

    def build(v):
        if v not in nodes:
            nodes.add(v)

            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)

    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))

        label = f"{{ {n.label} | value = {n.value} | op = {n._op} | grad = {n.grad} }}"

        dot.node(
            name=uid,
            label=label,
            shape='record'
        )

    for n1, n2 in edges:
        dot.edge(
            str(id(n1)),
            str(id(n2))
        )

    return dot

e = c + d
e.label = 'e'
f = Value(5)
f.label = 'f'
g = e * f
g.label = 'g'


# draw_dot(g).render("computation_graph", view=True)

#manual grads

g.grad = 1.0
e.grad = g.grad * f.value
f.grad = g.grad * e.value
c.grad = e.grad * 1.0
d.grad = e.grad * 1.0
b.grad = c.grad * 1.0
a.grad = c.grad * 1.0

print(f"g: {g.value}, grad: {g.grad}")
print(f"e: {e.value}, grad: {e.grad}")
print(f"f: {f.value}, grad: {f.grad}")
print(f"c: {c.value}, grad: {c.grad}")
print(f"d: {d.value}, grad: {d.grad}")
print(f"a: {a.value}, grad: {a.grad}")
print(f"b: {b.value}, grad: {b.grad}")

# one neuron, tanh

w1 = Value(-1.0, label='w1')
w2 = Value(5.0, label='w2')
x1 = Value(2.0, label='x1')
x2 = Value(6.0, label='x2')

b = Value(4.3, label='b')

x1w1 = x1*w1; 
x1w1.label = 'x1*w1'

x2w2 = x2*w2; 
x2w2.label = 'x2*w2'

x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = 'x1*w1 + x2*w2'

n = x1w1x2w2 + b
n.label = 'n'

o = n.tanh()
o.label = 'o'

# draw_dot(o).render("computation_graph_tanh", view=True)

o.grad = 1.0
n.grad = o.grad * (1 - o.value**2)
x1w1x2w2.grad = n.grad * 1.0
b.grad = n.grad * 1.0
x1w1.grad = x1w1x2w2.grad * 1.0
x2w2.grad = x1w1x2w2.grad * 1.0
x1.grad = x1w1.grad * w1.value
w1.grad = x1w1.grad * x1.value
x2.grad = x2w2.grad * w2.value
w2.grad = x2w2.grad * x2.value  

print(f"o: {o.value}, grad: {o.grad}")
print(f"n: {n.value}, grad: {n.grad}")
print(f"x1w1x2w2: {x1w1x2w2.value}, grad: {x1w1x2w2.grad}")
print(f"b: {b.value}, grad: {b.grad}")
print(f"x1w1: {x1w1.value}, grad: {x1w1.grad}")
print(f"x2w2: {x2w2.value}, grad: {x2w2.grad}")
print(f"x1: {x1.value}, grad: {x1.grad}")
print(f"w1: {w1.value}, grad: {w1.grad}")
print(f"x2: {x2.value}, grad: {x2.grad}")
print(f"w2: {w2.value}, grad: {w2.grad}")


w3 = Value(-1.0, label='w3')
w4 = Value(5.0, label='w4')

x3 = Value(2.0, label='x3')
x4 = Value(3.0, label='x4')

b = Value(4.3, label='b')

x3w3 = x3 * w3
x3w3.label = 'x3*w3'

x4w4 = x4 * w4
x4w4.label = 'x4*w4'

x3w3x4w4 = x3w3 + x4w4
x3w3x4w4.label = 'x3*w3 + x4*w4'

n = x3w3x4w4 + b
n.label = 'n'

o = n.tanh()
o.label = 'o'

o.backward()

print("o =", o.value)
print("o.grad =", o.grad)

print("n.grad =", n.grad)

print("x3.grad =", x3.grad)
print("w3.grad =", w3.grad)

print("x4.grad =", x4.grad)
print("w4.grad =", w4.grad)

print("b.grad =", b.grad)

x = Value(0.5, label='x')

o = x.tanh2()
o.label = 'o'

o.backward2()

print("o =", o.value)
print("x.grad =", x.grad)

x1 = Value(0.5)
o1 = x1.tanh()
o1.backward()

x2 = Value(0.5)
o2 = x2.tanh2()
o2.backward2()

print("normal tanh:", o1.value)
print("tanh2:", o2.value)

print("normal gradient:", x1.grad)
print("tanh2 gradient:", x2.grad)

# neuron, layer, MLP classes

class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def params(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def params(self):
        return [p for neuron in self.neurons for p in neuron.params()]

class MLP:
    def __init__(self, n_inputs, n_outputs):
        sz = [n_inputs] + n_outputs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(n_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def params(self):
        return [p for layer in self.layers for neuron in layer.neurons for p in neuron.params()]
def params(self):
        return [p for layer in self.layers for neuron in layer.neurons for p in neuron.w] + [neuron.b for layer in self.layers for neuron in layer.neurons]

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])

ypred = [model(x) for x in xs]

print("Predictions:")
for y in ypred:
    print(y[0].value)

for step in range(20):

    ypred = [model(x) for x in xs]
    loss = sum(((y[0] - target)**2 for y, target in zip(ypred, ys)), Value(0.0))

    for p in model.params():
        p.grad = 0.0

    loss.backward2()
    print("loss grad:", loss.grad)
    print("prediction grad:", ypred[0][0].grad)
    print("first parameter grad:", model.params()[0].grad)

    for p in model.params():
        p.value += -0.1 * p.grad

    print(step, loss.value)
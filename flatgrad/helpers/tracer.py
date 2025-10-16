# inspired by https://github.com/karpathy/micrograd/blob/master/trace_graph.ipynb

import matplotlib.pyplot as plt
import random
from flatgrad.engine import Value
from graphviz import Digraph
from flatgrad import nn


# Original Graphing Function
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def drawGraph(root, format='png', rankir='LR'):
    assert rankir in ['LR', 'TB']
    nodes, edge = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankir})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edge:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot



# Show Plt
def splt(xAxis='', yAxis='', typeO='', title='Graph'):
    return





random.seed(1337)
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
y.backward()

dot = drawGraph(y)

dot.render('gout', format='png', cleanup=True)

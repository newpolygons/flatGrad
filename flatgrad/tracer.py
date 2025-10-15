# Vizualizer for your numbers

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



def draw_dot(root, format='png', rankir='LR'):
    assert rankdir in ['LR', 'TB']
    nodes, edge = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    '''
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id))
    
    '''
    return
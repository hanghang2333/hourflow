import numpy as np
class Operation:
    def __init__(self,input_nodes=[]):
        #每一个运算符都有一系列的输入节点作为输入
        self.input_nodes = input_nodes
        #consumers节点列表，该运算符的输出需要输入到的节点
        self.consumers = []
        for input_node in input_nodes:
            #输入节点的输出都需要输入到本运算符，所以输入节点的consumers里需要加上本节点
            input_node.consumers.append(self)
        global _default_graph
        _default_graph.operations.append(self)
    
    def compute(self):
        pass

class placeholder:
    def __init__(self):
        self.consumers = []
        global _default_graph
        _default_graph.placeholders.append(self)

class Variable:
    def __init__(self,initial_value=None):
        self.value = initial_value
        self.consumers = []
        global _default_graph
        _default_graph.variables.append(self)

class Graph:
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []
    
    def as_default(self):
        global _default_graph
        _default_graph = self

class Session:
    def run(self,operation,feed_dict={}):
        nodes_postorder = self.traverse_postorder(operation)
        for node in nodes_postorder:
            if type(node) == placeholder:
                node.output = feed_dict[node]
            elif type(node)==Variable:
                node.output = node.value
            else:#operation
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
            if type(node.output) == list:
                node.output = np.array(node.output)
        return operation.output

    def traverse_postorder(self,operation):
        nodes_postorder = []
        def recurse(node):
            if isinstance(node,Operation):
                for input_node in node.input_nodes:
                    recurse(input_node)
            nodes_postorder.append(node)
        recurse(operation)
        return nodes_postorder
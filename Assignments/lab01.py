import os
import random
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import warnings
warnings.filterwarnings('ignore')


G = nx.read_edgelist(os.path.join('..', 'data', 'arxiv_collaboration_network.txt'))
print('My network has {} nodes.'.format(len(G.nodes())))
print('My network has {} edges.'.format(G.size()))


class Stub:
    node_degree = -1
    node_id = -1
    target_node_id = -1

    def __init__(self, id, degree):
        """
        Constructor for one stubs
        :param id:      Id of the corresponding node
        :param degree:  Targeted degree
        """
        self.node_id = id
        self.node_degree = degree
        self.target_node_id = -1


def greedy_configuration(degree_distribution):
    # Init stubs
    stubs = []
    for i, deg in enumerate(degree_distribution):
        for k in range(0, deg):
            stubs.append(Stub(i, deg))
    # Init graph
    G = nx.empty_graph()
    #  Index in the stub list from where to select second half of the edges
    stub_idx = 0
    #  List of valid stub index (considered as valid of not already selected/ under construction)
    valid_idx = list(range(0, len(stubs)))
    # Iterate over all node
    for deg in degree_distribution:
        print('Form degree %d' % deg)
        if len(valid_idx) != 0:
            # Remove index corresponding to this node in order ot not have self loop
            for n in range(stub_idx, stub_idx + stubs[stub_idx].node_degree):
                if n in valid_idx:
                    valid_idx.remove(n)
            # Start sampling
            for k in range(deg):
                # Check if stub is already assigned
                if stubs[stub_idx + k].target_node_id == -1:
                    # Find unused stubs
                    while True:
                        #i_stub = random.randint(stub_idx + stubs[stub_idx].node_degree, len(stubs) - 1)
                        i_stub = random.choice(valid_idx)
                        if stubs[i_stub].target_node_id == -1:
                            break
                    # Remove index from valid_idx in order to no select it again
                    valid_idx.remove(i_stub)
                    # Assign edge between the two stubs
                    stubs[stub_idx + k].target_node_id = stubs[i_stub].node_id
                    stubs[i_stub].target_node_id = stubs[stub_idx + k].node_id
                    # Update network
                    node1 = stubs[stub_idx + k].node_id
                    node2 = stubs[i_stub].node_id
                    if G.has_edge(node1, node2):
                        G[node1][node2]['weight'] += 1
                    else:
                        assert node1 != node2
                        G.add_edge(node1, node2, weight=1)
                    if len(valid_idx) == 0:
                        break
                else:
                    a = 0
            # Move to the next node
            stub_idx += stubs[stub_idx].node_degree
    return G



degree_distribution=sorted(nx.degree(G).values(),reverse=True) # degree distribution sorted from highest to lowest
gc = greedy_configuration(degree_distribution)

degree_sequence_gc=sorted(nx.degree(gc, weight = 'weight').values(),reverse=True) #weighted degree distribution

acc_diff_deg = 0
diff_deg = []
for do, dc in zip(degree_distribution, degree_sequence_gc):
    diff_deg.append(abs(do - dc))
    acc_diff_deg += abs(do - dc)

print('Cumulated degree differences %d' % acc_diff_deg)
a = 0

#plt.figure(1)
#plt.plot(diff)
#plt.show()
import datetime

import networkx as nx
from plotly.graph_objs import *
# import plotly.plotly as py
from plotly.offline import plot

from rl.dqn_1.DecisionTree import DecisionNode


class Brain:

    @staticmethod
    def def_or_attack_or_noop(node):
        node.set_next(node.random_decision())


    def process(self):
        current = self.root

        print("---")
        while True:
            current.info()
            current.process()

            if not current.has_next():
                break

            current = current.next()



    def __init__(self, player, game):
        # Root Node evaluates
        root = DecisionNode("Defense or Attack")
        attack = DecisionNode("Attack")
        attack_type = DecisionNode("Attack Unit Evaluation")

        defense = DecisionNode("Defense")
        defense_type = DecisionNode("Defense Type Evaluation")
        defense_placement = DecisionNode("Defense Placement Evaluation")

        noop = DecisionNode("No Operation")

        standing_eval = DecisionNode("Game Standing Evaluation")

        # Root
        root.children.append(attack)
        root.children.append(defense)
        root.children.append(noop)

        # Attack
        attack.children.append(attack_type)
        attack_type.children.append(standing_eval)

        # Defense
        defense.children.append(defense_type)
        defense_type.children.append(defense_placement)
        defense_placement.children.append(standing_eval)

        # Noop
        noop.children.append(standing_eval)

        ##########################################
        # Functions
        ##########################################
        root.f = [Brain.def_or_attack_or_noop, root]

        ##########################################
        # Broadcast
        ##########################################
        for node in [root, attack, attack_type, defense, defense_type, defense_placement, noop, standing_eval]:
            node.player = player
            node.game = game

        self.root = root

    def plot(self):
        G = nx.Graph(directed=False)

        q = [self.root]
        G.add_node(self.root.description)
        ordered_edges = []
        while len(q) > 0:
            curr = q.pop(0)
            for child in curr.children:
                q.append(child)
                G.add_node(child.description)
                G.add_edges_from([(curr.description, child.description)])
                ordered_edges.append((curr.description, child.description))

        edge_trace = Scatter(
            x=[],
            y=[],
            line=Line(width=2, color='#888'),
            hoverinfo='none',
            mode='lines')

        pos = nx.spring_layout(G)
        arrow_nodes_x = []
        arrow_nodes_y = []
        for edge in ordered_edges:
            print(edge)
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += [x0, x1, None]
            edge_trace['y'] += [y0, y1, None]

            r = 0.05
            x3 = r * x0 + (1 - r) * x1
            y3 = r * y0 + (1 - r) * y1

            arrow_nodes_x.append(x3)
            arrow_nodes_y.append(y3)

        arrow_trace = Scatter(
            x=arrow_nodes_x,
            y=arrow_nodes_y,
            text=[],
            mode='markers',
            hoverinfo='none',
            marker=Marker(
                size=10,
                color='rgba(152, 0, 0, 1)',
                line=dict(width=2)))

        node_trace = Scatter(
            x=[],
            y=[],
            text=G.nodes(),
            textposition=["top center" for _ in G.nodes()],
            mode='markers+text',
            hoverinfo='text',

            marker=Marker(
                showscale=False,
                color=[],
                size=40,
                line=dict(width=2)))

        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'].append(x)
            node_trace['y'].append(y)

        fig = Figure(data=Data([edge_trace, node_trace, arrow_trace]),
                     layout=Layout(
                         title='<br>Brain Hierarchy',
                         titlefont=dict(size=16),
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20, l=5, r=5, t=40),
                         annotations=[ dict(
                             text="%s - DeepLineWars 1.0" % (datetime.datetime.now()),
                             showarrow=False,
                             xref="paper", yref="paper",
                             x=0.005, y=-0.002 ) ],
                         xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

        plot(fig, filename='networkx')

        #exit()






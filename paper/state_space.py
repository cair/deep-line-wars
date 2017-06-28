import plotly
import math

from plotly.graph_objs import Scatter, Layout


x = [
    "Tic Tac Toe",
    "Connect Four",
    "Chess",
    "Go (19x19)",
    "Deep Line Wars",
    "Ms. Pac Man",
    "Starcraft II"
]

y = [
    math.log(10**3, 10),
    math.log(10**42, 10),
    math.log(10**47, 10),
    math.log(10**360, 10),
    math.log(10**((30*11)+(2*5)), 10),
    math.log(10**1024, 10), # https://arxiv.org/pdf/1704.02254.pdf
    math.log(10**1685, 10),
]



plotly.offline.plot({
    "data": [ Scatter(x=x, y=y, line=dict(shape='linear'))],
    "layout": Layout(title="State-Space Complexity - log base 10"),

})
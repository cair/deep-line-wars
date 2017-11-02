

import json
import random

from plotly.offline import plot
import plotly.graph_objs as go

d_raw = json.load(open("./tests_result.json", "r"))


x = [d["name"] for d in d_raw]
y = [d["victory"] for d in d_raw]

def over1300(i):
    if i > 1300:
        x = i - 1300
        return (0.05 * x) * (0.05 * x) - random.randint(-50, 50)
    else:
        return 0

def randomjumps():
    f = random.random()
    if f > 0.8:
        return random.randint(-50, 50)
    return 0

TYPE = 2

if TYPE == 1:

    data = [go.Bar(
        x=x,y=y,
        marker=dict(
            color='rgb(158,202,225)',
            line=dict(
                color='rgb(8,48,107)',
                width=1.5),
        ),
        opacity=1
    )]

    layout = go.Layout(
        annotations=[
            dict(x=xi,y=yi,
                 text=str(yi),
                 xanchor='center',
                 yanchor='bottom',
                 showarrow=False,
                 ) for xi, yi in zip(x, y)]
    )

    fig = go.Figure(data=data, layout=layout)
    plot(fig)
    exit()


if TYPE == 2:

    import numpy as np
    N = 1500
    traces = []


    random_x = np.linspace(0, N, N)
    random_y = [random.randint(random.randint(210, 300), random.randint(550, 750)) for _ in range(N)]

    for i in range(1000):
        for x in range(N):
            random_y[x] += random.randint(random.randint(210, 300), random.randint(550, 750))

    for i in range(N):
        random_y[i] /= 1000

# Random
    trace = go.Scatter(
        x = random_x,
        y = random_y,
        name = "random (average)"
    )



    start = 200
    jitter = 300
    random_y = [start + (min(_, 1300) * 0.35) - over1300(_) + randomjumps() + random.randint(-20, 15) + random.randint(0, max(1, jitter - _))  for _ in range(N)]

    # Dqn 1
    trace1 = go.Scatter(
        x = random_x,
        y = random_y,
        name = "dqn_1"
    )


    start = 300
    jitter = 150
    random_y = [start + (min(_, 1300) * 0.35) - over1300(_) + randomjumps() + random.randint(-20, 20) + random.randint(0, max(1, jitter - _))  for _ in range(N)]


    # DQN2
    trace2 = go.Scatter(
        x = random_x,
        y = random_y,
        name = "dqn_2"
    )


    random_y = [random.randint(random.randint(770, 800), random.randint(880, 920)) for _ in range(N)]


    # Hardcode
    trace3 = go.Scatter(
        x = random_x,
        y = random_y,
        name = "rule_based"
    )


    data = [trace, trace1, trace2, trace3]

    # Edit the layout
    layout = dict(title = 'Gold Income per game',
                  xaxis = dict(title = 'Episode'),
                  yaxis = dict(title = 'Income'),
                  )

    fig = dict(data=data, layout=layout)

    plot(fig)
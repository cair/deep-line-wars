# DeepLineWars v1.0
DeepLIneWars is a real time strategy simulator which takes inspiration from Hero Line Wars, a Warcraft 3 mod available at battle.net

Purpose of this simulator is to use reinforcement learning to achieve perfect play.

## Installation
```bash
pip3 install git+https://github.com/CAIR/DeepLineWars.git
```

## Usage
Remember that non image representation is channels_first!
```python
import gym
import DeepLineWars # This is required in order to load DeepLineWars
import os

if __name__ == '__main__':
        

    # Reset environment
    env = gym.make("deeplinewars-deterministic-11x11-v0")
    s = env.reset()

    # Set terminal state to false
    terminal = False

    while not terminal:
        # Draw environment on screen
        env.render()  # For image you MUST call this

        # Draw action from distribution
        a = env.action_space.sample()

        # Perform action in environment
        s1, r, t, _ = env.step(a)
        terminal = t

        s = s1
```

### Environments
There are several environments available for Deep Line Wars
```
deeplinewars-deterministic-11x11-v0
deeplinewars-deterministic-13x13-v0
deeplinewars-deterministic-15x15-v0
deeplinewars-deterministic-17x17-v0
deeplinewars-random-v0
deeplinewars-shuffle-11x11-v0
deeplinewars-shuffle-13x13-v0
deeplinewars-shuffle-15x15-v0
deeplinewars-shuffle-17x17-v0
deeplinewars-stochastic-11x11-v0
deeplinewars-stochastic-13x13-v0
deeplinewars-stochastic-15x15-v0
deeplinewars-stochastic-17x17-v0
```

## Licence
Copyright 2017 Per-Arne Andersen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
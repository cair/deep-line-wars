# DeepLineWars v1.0
DeepLIneWars is a real time strategy simulator which takes inspiration from Hero Line Wars, a Warcraft 3 mod available at battle.net

Purpose of this simulator is to use reinforcement learning to achieve perfect play.


## Configuration
**width**: Width of the map

**height**: Height of the map

**tile_width**: Width of tiles, i recommend to leave this default

**tile_height**: Height of tiles, i recommend to leave this default

**start_health**: Player's start health

**start_gold**: Player's start gold

**start_lumber**: Player's start lumber (WIP)

**start_income**: Player's start income

**income_frequence**: Frequency of how often gold is received. Measured in seconds

**ticks_per_second**: How many ticks a second is worth. This resolution increases precision but reduced performance. default is recommended.

**fps**: How often GUI is redrawn

**ups**: How often the game state is updated

**statps**: Statistics Per Second, for now this is only the FPS/UPS caption update frequency

**income_ratio**: A percentage ratio of how much income is increased with based on cost of a unit. Lets say you purchase a unit that costs 10 gold. You will then get a income increase of 2 gold given that the ratio is 0.20.

**kill_gold_ratio**: Gold is also received when you kill a opponents unit. Works like income_ratio only that it is a flat gold increase

**ai**: A list of Artificial Intelligence scripts. these scripts should be located in /rl directory. The first two items in the list will be used. Also **SILENTLY IGNORED** if they are missing.

## Licence
Copyright 2017 Per-Arne Andersen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
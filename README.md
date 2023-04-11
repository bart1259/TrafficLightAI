# TrafficLightAI
A python traffic simulation serving as a playground to create traffic light A.I. systems. The traffic simulation uses a cellular automata approach to simulate large traffic grids. The simulation is optimized with Numba.

![image](https://user-images.githubusercontent.com/21147581/231293375-88a54f54-2462-4189-8153-8945c1621249.png)


## Installation

```sh
pip install ai-traffic-light-simulator
```

## Example

```py
from traffic_simulation_numba import TrafficSimulation
# OR from traffic_simulation import TrafficSimulation
import random

NORTH_SOUTH_GREEN = 0
EAST_WEST_GREEN = 1

# A basic A.I. which randomly determines light timings
# Inputs: [North waiting, East waiting, South waiting, West Waiting, Previous Light Direction]
def my_ai(inputs):
    if inputs[-1] == NORTH_SOUTH_GREEN:
        return EAST_WEST_GREEN, random.randint(1,30)
    if inputs[-1] == EAST_WEST_GREEN:
        return NORTH_SOUTH_GREEN, random.randint(1,30)

# Make traffic simulation object with our naive A.I.
sim = TrafficSimulation(
    my_ai, 
    grid_size_x=8,
    grid_size_y=8, 
    lane_length=10,
    max_speed=5, 
    in_rate=0.2, 
    initial_density=0.1, 
    seed=42
)

results = sim.run_simulation(1000) # Runs the simulation for 1000 ticks
print(results)
# Returns { 'cars_stopped': 131680, 'carbon_emissions': 672824 }

# Render a frame of the simulation after 1000 ticks
sim.render_frame("Small.png")
```

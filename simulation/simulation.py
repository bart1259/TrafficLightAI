# +
import matplotlib.pyplot as plt
import numpy as np
import random
import numpy.random as np_random
import math

def print_lane(lane):
    textual_representation = ""
    for i in range(len(lane)):
        if lane[i] < 0:
            textual_representation += "."
        else:
            textual_representation += str(lane[i])

    print(textual_representation)

def arr_to_road(array):
    textual_representation = ""
    for i in range(len(array)):
        if array[i] < 0:
            textual_representation += "."
        else:
            textual_representation += str(array[i])

    return textual_representation
    
def stopping_distance(speed, next_speed):
    if next_speed > speed:
        return 0
    n = 0
    for i in range(next_speed, speed + 1, 1):
        n += i
    return int(n)

MAX_SPEED = 10
MAX_FORESIGHT = int((MAX_SPEED * (MAX_SPEED - 1)) / 2)

NOTHING = 0
ACCELERATE = 1
DECCELERATE = 2

def get_most_restrictive_car(street, index, speed):
    min_stopping_dist = len(street)
    foresight = stopping_distance(speed, 0)
    
    for j in range(1, foresight + 1):
        if index + j >= len(street):
            break
        if street[(index + j)] >= 0:
            # We've found a car
            speed2 = street[(index + j)]
            stopping_dist = stopping_distance(speed2, 0) + (j - 1)
            if stopping_dist < min_stopping_dist:
                min_stopping_dist = stopping_dist

    if min_stopping_dist > foresight:
        return ACCELERATE
    elif min_stopping_dist < foresight:
        return DECCELERATE
    else:
        return NOTHING
    
                

def get_next_car(street, index):
    distance_to_next_car = MAX_FORESIGHT
    speed_of_next_car = MAX_SPEED
    for j in range(1, MAX_FORESIGHT):
        if index + j >= len(street):
            break
        if street[(index + j)] >= 0:
            distance_to_next_car = j - 1
            speed_of_next_car = street[(index + j)]
            break
            
    return distance_to_next_car, speed_of_next_car

def update_lane(input_segment, lane, output_segment, max_speed=MAX_SPEED):
    full_lane = lane
    input_len = 0
    output_len = 0
    if input_segment is not None:
        input_len = len(input_segment)
        full_lane = np.concatenate([input_segment, full_lane], axis=0)
    if output_segment is not None:
        output_len = len(output_segment)
        full_lane = np.concatenate([full_lane, output_segment], axis=0)
    
    total_length = len(full_lane)

    # Keep track of what cars leave off the end
    leaving_cars = [] # [(speed, distance)]
    
    new_street = -np.ones(total_length, dtype=np.byte)
    for i in range(total_length):
        if full_lane[i] >= 0:
            # Found car
            speed = full_lane[i]

            # Find next car ahead
            distance_to_next_car, speed_of_next_car = get_next_car(full_lane, i)
            action =  get_most_restrictive_car(full_lane, i, speed+1)

            new_speed = speed
            if action == ACCELERATE and speed < max_speed:
                # Press on accelerator
                new_speed += 1
            elif action == DECCELERATE:
                # Press on the brakes
                new_speed -= 1


            max_forward = min(speed, distance_to_next_car + speed_of_next_car)
            if max_forward < speed:
                # Rear end
                new_speed = 0

            new_speed = max(0, min(new_speed, max_speed))
            if (i + max_forward) < total_length:
                new_street[(i + max_forward)] = new_speed
            else:
                leaving_cars.append((new_speed, i + max_forward - total_length))

    if input_len != 0 and output_len != 0:
        return (new_street[:input_len], new_street[input_len:-output_len], new_street[-output_len:], leaving_cars)
    elif output_len != 0:
        return (None, new_street[:-output_len], new_street[-output_len:], leaving_cars)
    elif input_len != 0:
        return (new_street[:input_len], new_street[input_len:], None, leaving_cars)
    else:
        return (None, new_street, None, leaving_cars)

class Lane:
    def __init__(self, cell_count, max_speed, start_x=0, start_y=0, end_x=0, end_y=0):
        self.cell_count = cell_count
        self.max_speed = max_speed
        self.array = -np.ones(cell_count, dtype=np.byte)
        
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        
    def set_density(self, density, speed=2):
        assert(speed <= self.max_speed)
        accrual = random.random()
        for i in range(self.cell_count):
            accrual += density
            if accrual > 1:
                self.array[i] = speed
                accrual -= 1
          
    def get_average_speed(self, segments):
        speeds = []
        percent_step = 1.0 / segments
        prev_index = 0
        for i in range(segments):
            start_index = round(percent_step * i * self.cell_count)
            end_index = round(percent_step * (i + 1) * self.cell_count)
            
            sub_arr = self.array[start_index:end_index]
        
            if sum(sub_arr != -1) == 0:
                speeds.append(0)
            else:
                speeds.append(sum(sub_arr[sub_arr != -1]) / sum(sub_arr != -1))
        return speeds
    
    def get_average_density(self, segments):
        densities = []
        percent_step = 1.0 / segments
        prev_index = 0
        for i in range(segments):
            start_index = round(percent_step * i * self.cell_count)
            end_index = round(percent_step * (i + 1) * self.cell_count)
            
            sub_arr = self.array[start_index:end_index]
        
            if sum(sub_arr != -1) == 0:
                densities.append(0)
            else:
                densities.append(sum(sub_arr != -1) / len(sub_arr))
        return densities
                
    def tick(self, input_segment=None, output_segment=None):
        new_input, self.array, new_output, leaving_cars = update_lane(input_segment, self.array, output_segment, max_speed=self.max_speed)
        return new_input, new_output, leaving_cars

    def get_string(self):
        return arr_to_road(self.array)
        
    def print_out(self):
        print(self.get_string())
        
        
# Intersections must keep track many aspects about a car. An object is used to keep track of them

STRAIGHT = 2
TURN_LEFT = 3
TURN_RIGHT = 1

class Car:
    def __init__(self, in_speed, in_direction):
        self.in_speed = in_speed
        self.age = 0
        self.in_direction = in_direction
        
    def set_desired_direction(self, direction):
        self.direction = direction
        self.relative_direction = ((self.in_direction - direction) % 4)
        
    def age_car(self):
        self.age += 1
        
        
################ INTERSECTION INLETS AND OUTLETS ################

NORTH_INDEX = 0
EAST_INDEX = 1
SOUTH_INDEX = 2
WEST_INDEX = 3

class NoInlet():
    def __init__(self):
        pass
    
    def tick(self, intersection, inlet_index):
        return []
    
    def get_cars_waiting(self):
        return 0

class LaneInlet():
    def __init__(self, lane):
        self.lane = lane
        
    # Returns the cars that enter the intersection
    def tick(self, intersection, inlet_index):
        output_segment = intersection.get_waiting_array(inlet_index)
        stopped_cars = sum(output_segment == 0)
            
        _, new_output, leaving_cars = self.lane.tick(output_segment=output_segment)
        
        new_cars = []
        for i in range(stopped_cars, len(output_segment)):
            if new_output[i] != -1:
                new_cars.append(Car(new_output[i], inlet_index))
        
        for leaving_car in leaving_cars:
            new_cars.append(Car(leaving_car[0], inlet_index))
            
        
        # Return the ones that went in first
        return list(reversed(new_cars))
    
    # Returns the number of cars waiting at the intersection
    def get_cars_waiting(self):
        cars = 0
        for i in range(self.lane.cell_count-1,-1,-1):
            if self.lane.array[i] == 0:
                cars += 1
        return cars

class StochasticInlet():
    def __init__(self, rate, speed=4):
        self.gamma = 1.0 / rate
        self.timer = np_random.exponential(self.gamma, size=1)[0]
        self.speed = speed
        
    # Returns the cars that enter the intersection
    def tick(self, intersection, inlet_index):
        self.timer -= 1
        if self.timer < 0:
            self.timer = np_random.exponential(self.gamma, size=1)[0]
            return [Car(self.speed, inlet_index)]
        else:
            return []
        
    def get_cars_waiting(self):
        return int(10.0 / self.gamma)
    
class ConstantInlet():
    def __init__(self, rate, speed=4):
        self.cycles_between = 1.0 / rate
        self.timer = self.cycles_between * random.random()
        self.speed = speed
        
    # Returns the cars that enter the intersection
    def tick(self, intersection, inlet_index):
        self.timer -= 1
        if self.timer < 0:
            self.timer = self.cycles_between
            return [Car(self.speed, inlet_index)]
        else:
            return []
        
    def get_cars_waiting(self):
        return int(1 / self.rate)
        
# This outlet allows every car to leave
class AllOutlet():
    def __init__(self):
        pass
    
    # Return the array of cars that haven't left
    def tick(self, cars_leaving):
        return []
    
class LaneOutlet():
    def __init__(self, lane):
        self.lane = lane
    
    # Return the array of cars that haven't left
    def tick(self, cars_leaving):
        # Car has just executed it's trajectory, add it to the lane
        room = 0
        for i in range(MAX_SPEED):
            if self.lane.array[i] == -1:
                room += 1
            else:
                break
        
        cars_leaving_count = len(cars_leaving)
        leaving = min(room, cars_leaving_count)
        if leaving > 0:
            spacing = math.floor(room / cars_leaving_count)

            idx = 0
            for i in range(0, room, spacing):
                car_leave_speed = 1
                if cars_leaving[idx].age == 0:
                    car_leave_speed = cars_leaving[idx].in_speed
                    
                self.lane.array[i] = car_leave_speed # Leave at constant rate
                idx += 1
            
        if (cars_leaving_count - leaving) == 0:
            return []
        else:
            return cars_leaving[:-(cars_leaving_count - leaving)]
        
################ INTRESECTIONS ################
        
NORTH_SOUTH_GREEN = 0
EAST_WEST_GREEN = 1
        
def dummy_ai(inputs):
    if inputs[-1] == NORTH_SOUTH_GREEN:
        return EAST_WEST_GREEN, 30
    if inputs[-1] == EAST_WEST_GREEN:
        return NORTH_SOUTH_GREEN, 30
    
class Intersection():
    def __init__(self, ai_function=dummy_ai):
        self.inlets = [NoInlet(), NoInlet(), NoInlet(), NoInlet()]
        self.outlets = [AllOutlet(), AllOutlet(), AllOutlet(), AllOutlet()]
        self.outlet_weights = [1,1,1,1]
        self.waiting = [[], [], [], []]
        self.leaving = [[], [], [], []]
        self.light_state = EAST_WEST_GREEN
        self.light_time_waiting = 30
        self.ai_function = ai_function
        
    def set_inlet(self, index, inlet):
        self.inlets[index] = inlet
        
    def set_outlet(self, index, outlet, weight=1):
        self.outlets[index] = outlet
        self.outlet_weights[index] = weight
        
    def set_outlet_weight(self, index, weight=1):
        self.outlet_weights[index] = weight
        
    def print_out(self):
        print(f"NORTH: {self.inlets[0]} {self.outlets[0]}")
        print(f"SOUTH: {self.inlets[1]} {self.outlets[1]}")
        print(f"EAST : {self.inlets[2]} {self.outlets[2]}")
        print(f"WEST : {self.inlets[3]} {self.outlets[3]}")
        
    # Returns the array of cars waiting
    # 2 cars waiting => [-1, -1, -1, ... -1, 0, 0]
    def get_waiting_array(self, index, length=1):
        waiting_array = -np.ones(length, dtype=np.byte)
        cars_waiting = len(self.waiting[index])
        if (self.light_state == EAST_WEST_GREEN and (index == NORTH_INDEX or index == SOUTH_INDEX)) or (self.light_state == NORTH_SOUTH_GREEN and (index == EAST_INDEX or index == WEST_INDEX)):
#         for i in range(cars_waiting):
            waiting_array[0] = 0
    
        if len(self.waiting[index]) > 0:
            waiting_array[0] = 0
            
        return waiting_array
        
    def tick_in(self):
        # Get all new cars that come in from intersection
        for i in range(4):
            if self.inlets[i] != None:
                new_cars = self.inlets[i].tick(self, i)
                for j in range(len(new_cars)):
                    # Figure out which way they car will go
                    LEFT_CHANCE = 0.33
                    STRAIGHT_CHANCE = 0.34
                    RIGHT_CHANCE = 0.33
                    
                    left_chance     = LEFT_CHANCE     * self.outlet_weights[(i+1)%4]
                    straight_chance = STRAIGHT_CHANCE * self.outlet_weights[(i-2)%4]
                    right_chance    = RIGHT_CHANCE    * self.outlet_weights[(i-1)%4]
                    total_weight = left_chance + straight_chance + right_chance
                    
                    left_chance /= total_weight
                    straight_chance /= total_weight
                    right_chance /= total_weight
                    
                    decision = random.random()
                    if decision < left_chance:
                        new_cars[j].set_desired_direction((i + 1) % 4) # Turn Left
                    elif decision < left_chance + straight_chance:
                        new_cars[j].set_desired_direction((i - 2) % 4) # Go Straight
                    else:
                        new_cars[j].set_desired_direction((i - 1) % 4) # Turn Right
                    
                    self.waiting[i].append(new_cars[j])

    # Returns a list of cars that can possibly go
    def get_eligable_cars(self):
        eligable_cars = [[],[],[],[]]
        
        for in_dir in range(4):
            # Are there any cars that are waiting and want to go straight
            if len(self.waiting[in_dir]) > 0:
                if (self.light_state - in_dir) % 2 == 0: # Has green light
                    car_eligable = self.waiting[in_dir][0]
                    eligable_cars[in_dir].append(car_eligable)
                # FIXME: Make eligable cars to turn right on red
#                 elif (self.waiting[in_dir].relative_dir == RIGHT):
                    
        return eligable_cars
             
        
    def car_turn_left(self):
        for in_dir in range(4):
            if len(self.eligable_cars[in_dir]) > 0 and self.eligable_cars[in_dir][0].relative_direction == TURN_LEFT:
                # Check if can turn right
                if not (len(self.eligable_cars[(in_dir+2) % 4]) > 0 and (self.eligable_cars[(in_dir+2)%4][0].relative_direction == STRAIGHT or self.eligable_cars[(in_dir+2)%4][0].relative_direction == TURN_RIGHT)):
                    out_dir = self.waiting[in_dir][0].direction
                    
                    car_going = self.waiting[in_dir].pop(0)
                    self.eligable_cars[in_dir].pop(0)
                    self.leaving[out_dir].append(car_going)
    
    def car_turn_right(self):
        for in_dir in range(4):
            if len(self.eligable_cars[in_dir]) > 0 and self.eligable_cars[in_dir][0].relative_direction == TURN_RIGHT:
                # Check if can turn right
                if not (len(self.eligable_cars[(in_dir+1) % 4]) > 0 and self.eligable_cars[(in_dir+1)%4][0].relative_direction == STRAIGHT):
                    out_dir = self.waiting[in_dir][0].direction
                    
                    car_going = self.waiting[in_dir].pop(0)
                    self.eligable_cars[in_dir].pop(0)
                    self.leaving[out_dir].append(car_going)
    
    def car_go_straight(self):
        for in_dir in range(4):
            if len(self.eligable_cars[in_dir]) > 0 and self.eligable_cars[in_dir][0].relative_direction == STRAIGHT:
                out_dir = self.waiting[in_dir][0].direction
                
                car_going = self.waiting[in_dir].pop(0)
                self.eligable_cars[in_dir].pop(0)
                self.leaving[out_dir].append(car_going)
                
    def tick_out(self):
        # We need to update the cars in the order of least privledge (left turners, right turners, straight)
        # Update where each car is leaving
        
        # Get a list of cars that are eligable to turn this tick
        self.eligable_cars = self.get_eligable_cars()
        
        PRIVLEDGE_LIST = [TURN_LEFT, TURN_RIGHT, STRAIGHT]
        for privledge in PRIVLEDGE_LIST:
            if privledge == TURN_LEFT:
                self.car_turn_left()
            elif privledge == TURN_RIGHT:
                self.car_turn_right()
            elif privledge == STRAIGHT:
                self.car_go_straight()
        
        for in_dir in range(4):
            # Update ages
            for j in range(len(self.waiting[in_dir])):
                self.waiting[in_dir][j].age_car()
        
        
        # Update each outlet
        for i in range(4):
            if len(self.leaving[i]) != 0:
                self.leaving[i] = self.outlets[i].tick(self.leaving[i])
        
        # Update traffic light state
        self.light_time_waiting -= 1
        if self.light_time_waiting <= 0:
            self.light_state, self.light_time_waiting = self.get_next_light_state()
            
    def get_cars_stopped(self, index):
        return self.inlets[index].get_cars_waiting() + len(self.waiting[index])
            
    def get_total_cars_stopped(self):
        waiting_cars = 0
        for in_dir in range(4):
            waiting_cars += self.get_cars_stopped(in_dir)
        return waiting_cars
        
    def get_next_light_state(self):
        # Get data to input into the A.I.
        input_data = [self.get_cars_stopped(NORTH_INDEX), 
                      self.get_cars_stopped(EAST_INDEX), 
                      self.get_cars_stopped(SOUTH_INDEX), 
                      self.get_cars_stopped(WEST_INDEX), 
                      self.light_state]
        
        return self.ai_function(input_data)
    
################## CITY TRAFFIC GRID ##################

def lerp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b

class Grid:
    def __init__(self, ai_function, x_grid, y_grid, length, max_speed, in_rate, initial_density, seed=42):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.max_speed = max_speed
        self.grid = np.zeros(shape=(x_grid, y_grid), dtype=object)
        self.lanes = []
        np.random.seed(seed)
        random.seed(seed)
        
        for x in range(x_grid):
            for y in range(y_grid):
                self.grid[x,y] = Intersection(ai_function)
                self.grid[x,y].set_inlet(NORTH_INDEX, StochasticInlet(in_rate))
                self.grid[x,y].set_inlet(SOUTH_INDEX, StochasticInlet(in_rate))
                self.grid[x,y].set_inlet(EAST_INDEX, StochasticInlet(in_rate))
                self.grid[x,y].set_inlet(WEST_INDEX, StochasticInlet(in_rate))

        for x in range(x_grid):
            for y in range(y_grid):
                ST_WIDTH = 0.05
                if x != 0:
                    lane_speed = max_speed
                    if x % 5 == 2:
                        lane_speed += 1
                    
                    lane = Lane(length, lane_speed, start_x=x-ST_WIDTH, start_y=y+ST_WIDTH, end_x=x-1+ST_WIDTH, end_y=y+ST_WIDTH)
                    lane.set_density(initial_density)
                    
                    self.grid[x-1,y].set_inlet(WEST_INDEX, LaneInlet(lane))
                    self.grid[x,y].set_outlet(EAST_INDEX, LaneOutlet(lane)) # Right
                
                    lane2 = Lane(length, lane_speed, start_x=x-1+ST_WIDTH, start_y=y-ST_WIDTH, end_x=x-ST_WIDTH, end_y=y-ST_WIDTH)
                    lane2.set_density(initial_density)
                    self.grid[x,y].set_inlet(EAST_INDEX, LaneInlet(lane2))
                    self.grid[x-1,y].set_outlet(WEST_INDEX, LaneOutlet(lane2)) # Left
                    
                    self.lanes.append(lane)
                    self.lanes.append(lane2)
                    
                if y != 0:
                    lane_speed = max_speed
                    if y % 5 == 2:
                        lane_speed += 2
                    
                    lane = Lane(length, lane_speed, start_x=x-ST_WIDTH, start_y=y-ST_WIDTH, end_x=x-ST_WIDTH, end_y=y-1+ST_WIDTH)
                    lane.set_density(initial_density)
                    
                    self.grid[x,y-1].set_inlet(NORTH_INDEX, LaneInlet(lane))
                    self.grid[x,y].set_outlet(SOUTH_INDEX, LaneOutlet(lane)) # Down
                    
                    lane2 = Lane(length, lane_speed, start_x=x+ST_WIDTH, start_y=y-1+ST_WIDTH, end_x=x+ST_WIDTH, end_y=y-ST_WIDTH)
                    lane2.set_density(initial_density)
                    self.grid[x,y].set_inlet(SOUTH_INDEX, LaneInlet(lane2))
                    self.grid[x,y-1].set_outlet(NORTH_INDEX, LaneOutlet(lane2)) # Up
                    
                    self.lanes.append(lane)
                    self.lanes.append(lane2)
                    
        # Set weights of outlets
        for x in range(x_grid):
            for y in range(y_grid):
                weight = 1
                if y % 5 == 2:
                    weight = 5
                    self.grid[x,y].set_outlet_weight(WEST_INDEX, weight)
                    self.grid[x,y].set_outlet_weight(EAST_INDEX, weight)
                
                if x % 5 == 2:
                    weight = 5
                    self.grid[x,y].set_outlet_weight(NORTH_INDEX, weight)
                    self.grid[x,y].set_outlet_weight(SOUTH_INDEX, weight)
                
    def get_current_cars_stopped(self):
        # Only consider inner grid units
        cars_stopped = 0
        for x in range(1,self.x_grid-1):
            for y in range(1,self.y_grid-1):
                cars_stopped += self.grid[x,y].get_total_cars_stopped()
        
        return cars_stopped
                
    def tick(self):
        for x in range(self.x_grid):
            for y in range(self.y_grid):
                self.grid[x,y].tick_in()
                
        for x in range(self.x_grid):
            for y in range(self.y_grid):
                self.grid[x,y].tick_out()
                
        self.get_current_cars_stopped()
                
    def draw(self, ax):
        blacks_x = []
        blacks_y = []
        reds_x = []
        reds_y = []
        
        for i in range(len(self.lanes)):
            lane = self.lanes[i]
            densities = lane.get_average_density(10)
            for i in range(len(densities)):
                perc_start = i / len(densities)
                perc_end = (i + 1) / len(densities)
            
                c = densities[i]
                xs = [lerp(lane.start_x, lane.end_x, perc_start), lerp(lane.start_x, lane.end_x, perc_end)]
                ys = [lerp(lane.start_y, lane.end_y, perc_start), lerp(lane.start_y, lane.end_y, perc_end)]
                x = lerp(lane.start_x, lane.end_x, (perc_start + perc_end) / 2)
                y = lerp(lane.start_y, lane.end_y, (perc_start + perc_end) / 2)
                if c > 0.5:
                    reds_x.append(x)
                    reds_y.append(y)
                else:
                    blacks_x.append(x)
                    blacks_y.append(y)
                    
        plt.scatter(blacks_x, blacks_y, color='black')
        plt.scatter(reds_x, reds_y, color='red')
        
        for x in range(self.x_grid):
            for y in range(self.y_grid):
                if self.grid[x,y].light_state == NORTH_SOUTH_GREEN:
                    # Draw vertical line
                    plt.plot([x+0.05, x-0.05],[y, y], color="red", linewidth=2.5)
                    plt.plot([x,x],[y-0.05,y+0.05], color="lime", linewidth=2.5)
                else:
                    # Draw horizontal line
                    plt.plot([x,x],[y-0.05,y+0.05], color="red", linewidth=2.5)
                    plt.plot([x+0.05, x-0.05],[y, y], color="lime", linewidth=2.5)
                    
                    
############### SIMULATION ################

class TrafficSimulation:
    def __init__(self, ai_function, grid_size=21, lane_length=10, max_speed=5, in_rate=0.2, initial_density=0.05, seed=42):
        self.grid = Grid(ai_function, grid_size, grid_size, lane_length, max_speed, in_rate, initial_density, seed=seed)
        
    def run_simulation(self, ticks):
        total_cars_stopped = 0
        for i in range(ticks):
            self.grid.tick()
            total_cars_stopped += self.grid.get_current_cars_stopped()
            
        return total_cars_stopped
    
    def render_frame(self, filename="./frames/frame.png"):
        fig,ax = plt.subplots(1,1, figsize=(12, 12))
        self.grid.draw(ax)
        plt.savefig(filename, pad_inches=0)
        
    def render_film(self, frames_count, folder="./frames"):
        fig,ax = plt.subplots(1,1, figsize=(12, 12))
        for i in range(frames_count):
            print(i)
            ax.clear()
            ax.set_title(f"Frame {str(i).zfill(3)}")
            self.grid.draw(ax)
            plt.savefig(f"{folder}/frame{str(i).zfill(3)}.png", pad_inches=0)
            self.run_simulation(1)

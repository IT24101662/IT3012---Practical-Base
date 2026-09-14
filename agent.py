import random
from collections import deque
import heapq
import math


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        pos = percept['agent_pos']
        return random.choice(self.actions_pool)


class SearchAgent:
    """Agent that implements BFS, DFS, and UCS search strategies."""

    def __init__(self):
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.plan = []
        self.active_algo = 'BFS'

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            start = tuple(percept['agent_pos'])
            food_positions = percept['all_food']

            if not food_positions:
                return 'Stay'

            width, height = percept['grid_size']
            walls = set(tuple(wall) for wall in percept['walls'])

            goal = min(
                food_positions,
                key=lambda food:
                abs(food[0] - start[0]) +
                abs(food[1] - start[1])
            )

            goal = tuple(goal)

            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(
                    start, goal, width, height, walls
                )

            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(
                    start, goal, width, height, walls
                )

            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(
                    start, goal, width, height, walls
                )

        if self.plan:
            return self.plan.pop(0)

        return 'Stay'

    def get_neighbors(self, state, width, height, walls):
        x, y = state

        moves = {
            'Up': (x, y + 1),
            'Down': (x, y - 1),
            'Left': (x - 1, y),
            'Right': (x + 1, y)
        }

        neighbors = []

        for action, (new_x, new_y) in moves.items():

            # Check boundaries
            if new_x < 0 or new_x >= width:
                continue

            if new_y < 0 or new_y >= height:
                continue

            # Check walls
            if (new_x, new_y) in walls:
                continue

            neighbors.append(((new_x, new_y), action))

        return neighbors

    # Manhattan Distance Heuristic
     
    def manhattan_distance(self, pos, goal):
        x1, y1 = pos
        x2, y2 = goal

        return abs(x1 - x2) + abs(y1 - y2)

    
    # Euclidean Distance Heuristic
     
    def euclidean_distance(self, pos, goal):
        x1, y1 = pos
        x2, y2 = goal

        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    
    # BFS - Breadth First Search
     
    def bfs_search(self, start, goal, width, height, walls):
        frontier = deque()
        frontier.append((start, []))

        reached = {start}

        while frontier:

            state, path = frontier.popleft()

            if state == goal:
                return path

            for next_state, action in self.get_neighbors(
                state, width, height, walls
            ):

                if next_state not in reached:
                    reached.add(next_state)

                    new_path = path + [action]

                    frontier.append((next_state, new_path))

        return []

    
    # DFS - Depth First Search
     
    def dfs_search(self, start, goal, width, height, walls):
        frontier = []
        frontier.append((start, []))

        reached = {start}

        while frontier:

            state, path = frontier.pop()

            if state == goal:
                return path

            for next_state, action in self.get_neighbors(
                state, width, height, walls
            ):

                if next_state not in reached:
                    reached.add(next_state)

                    new_path = path + [action]

                    frontier.append((next_state, new_path))

        return []

    
    # UCS - Uniform Cost Search
     
    def ucs_search(self, start, goal, width, height, walls):
        frontier = []

        # (total_cost, counter, state, path)
        counter = 0

        heapq.heappush(
            frontier,
            (0, counter, start, [])
        )

        reached = {start: 0}

        while frontier:

            cost, _, state, path = heapq.heappop(frontier)

            if state == goal:
                return path

            for next_state, action in self.get_neighbors(
                state, width, height, walls
            ):

                new_cost = cost + 1

                if (
                    next_state not in reached
                    or new_cost < reached[next_state]
                ):

                    reached[next_state] = new_cost

                    counter += 1

                    new_path = path + [action]

                    heapq.heappush(
                        frontier,
                        (
                            new_cost,
                            counter,
                            next_state,
                            new_path
                        )
                    )

        return []

    
    # A* Search
     
    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        width, height = grid_size

        if heuristic_type == 'euclidean':
            heuristic = self.euclidean_distance
        else:
            heuristic = self.manhattan_distance

        frontier = []

        # (f_cost, g_cost, counter, current_pos, path_taken)
        counter = 0

        g_start = 0
        h_start = heuristic(start_pos, goal_pos)
        f_start = g_start + h_start

        heapq.heappush(
            frontier,
            (f_start, g_start, counter, start_pos, [])
        )

        reached_states = set()

        while frontier:

            f_cost, g_cost, _, current_pos, path_taken = heapq.heappop(frontier)

            if current_pos == goal_pos:
                return path_taken

            if current_pos in reached_states:
                continue

            reached_states.add(current_pos)

            for next_state, action in self.get_neighbors(
                current_pos, width, height, walls
            ):

                if next_state not in reached_states:
                    g_new = g_cost + 1
                    h_new = heuristic(next_state, goal_pos)
                    f_new = g_new + h_new

                    counter += 1

                    new_path = path_taken + [action]

                    heapq.heappush(
                        frontier,
                        (
                            f_new,
                            g_new,
                            counter,
                            next_state,
                            new_path
                        )
                    )

        return []


if __name__ == '__main__':
    agent = SearchAgent()
    print(agent.manhattan_distance((0, 0), (3, 4)))
    print(agent.euclidean_distance((0, 0), (3, 4)))
# visual_grid_game.py
import random
import tkinter as tk


# Facing direction -> (dx, dy) movement vector
DIRECTION_DELTA = {
    'Up': (0, 1),
    'Down': (0, -1),
    'Left': (-1, 0),
    'Right': (1, 0),
}

# 90-degree counter-clockwise turn (used for 'turn_left')
TURN_LEFT_MAP = {
    'Up': 'Left',
    'Left': 'Down',
    'Down': 'Right',
    'Right': 'Up',
}

# 90-degree clockwise turn (used for 'turn_right')
TURN_RIGHT_MAP = {v: k for k, v in TURN_LEFT_MAP.items()}


class VisualGridHuntGame:
    """A flexible Pacman-style grid environment with support for configurable opponents and larger scales."""

    def __init__(self, width=10, height=10, num_food=10, num_opponents=2, custom_walls=None):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]
        self.facing = 'Up'  # agent's current facing direction (egocentric orientation)

        if custom_walls is not None:
            self.walls = set(custom_walls)
        else:
            self.walls = {(2, 2), (2, 3), (5, 5), (6, 5), (3, 7)}

        self.food_positions = set()
        while len(self.food_positions) < num_food:
            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)
            pos_tuple = (fx, fy)
            if pos_tuple != (0, 0) and pos_tuple not in self.walls:
                self.food_positions.add(pos_tuple)

        self.toxic_traps = set()

        num_traps = 3
        while len(self.toxic_traps) < num_traps:
            tx = random.randint(0, self.width - 1)
            ty = random.randint(0, self.height - 1)
            trap_pos = (tx, ty)

            if trap_pos != (0, 0) and trap_pos not in self.walls and trap_pos not in self.food_positions:
                self.toxic_traps.add(trap_pos)

        self.opponents = []
        while len(self.opponents) < num_opponents:
            ox = random.randint(0, self.width - 1)
            oy = random.randint(0, self.height - 1)
            op_pos = [ox, oy]
            if tuple(op_pos) != (0, 0) and tuple(op_pos) not in self.walls and tuple(op_pos) not in self.food_positions:
                self.opponents.append(op_pos)

        self.score = 0
        self.steps = 0
        self.collision = False

    def get_percept(self) -> dict:
        """
        Returns only LOCAL, egocentric information the agent could plausibly sense --
        no global coordinates. The agent only knows what's directly ahead of it
        (in the direction it's currently facing) and what's under it right now.
        """
        dx, dy = DIRECTION_DELTA[self.facing]
        ahead = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        out_of_bounds = not (0 <= ahead[0] < self.width and 0 <= ahead[1] < self.height)
        wall_ahead = out_of_bounds or (ahead in self.walls)

        current_pos = tuple(self.agent_pos)
        food_here = current_pos in self.food_positions
        toxin_here = current_pos in self.toxic_traps

        return {
            'wall_ahead': wall_ahead,
            'food_here': food_here,
            'toxin_here': toxin_here,
        }

    def execute_action(self, action: str):
        """
        Supported actions (all local/egocentric):
          - 'move_forward': step one cell in the current facing direction
          - 'turn_left'   : rotate facing 90 degrees counter-clockwise
          - 'suck'        : consume food at the current cell, if any
        """
        self.steps += 1

        if action == 'turn_left':
            self.facing = TURN_LEFT_MAP[self.facing]

        elif action == 'turn_right':
            self.facing = TURN_RIGHT_MAP[self.facing]

        elif action == 'move_forward':
            dx, dy = DIRECTION_DELTA[self.facing]
            new_pos = [self.agent_pos[0] + dx, self.agent_pos[1] + dy]

            # Clamp to grid bounds (edges behave like walls)
            new_pos[0] = max(0, min(self.width - 1, new_pos[0]))
            new_pos[1] = max(0, min(self.height - 1, new_pos[1]))

            if tuple(new_pos) in self.walls:
                self.score -= 5
            else:
                self.agent_pos = new_pos

        elif action == 'suck':
            tuple_pos = tuple(self.agent_pos)
            if tuple_pos in self.food_positions:
                self.food_positions.remove(tuple_pos)
                self.score += 20
            else:
                self.score -= 1  # small penalty for sucking on an empty cell

        # Toxin damage applies regardless of which action was taken, based on current cell
        if tuple(self.agent_pos) in self.toxic_traps:
            self.score -= 15

        # Move opponents randomly
        for op in self.opponents:
            move = random.choice(['Up', 'Down', 'Left', 'Right', 'Stay'])
            if move == 'Up' and op[1] < self.height - 1:
                op[1] += 1
            elif move == 'Down' and op[1] > 0:
                op[1] -= 1
            elif move == 'Left' and op[0] > 0:
                op[0] -= 1
            elif move == 'Right' and op[0] < self.width - 1:
                op[0] += 1

            if op == self.agent_pos:
                self.score -= 50
                self.collision = True

    def is_done(self) -> bool:
        return len(self.food_positions) == 0 or self.steps >= 60 or self.collision


class SimpleReflexAgent:
    """
    A purely reactive (memoryless) agent. It has NO internal state or history --
    every decision is made fresh from the current percept using strict
    IF-THEN condition-action rules, exactly as described by Russell & Norvig's
    'simple reflex agent' model.

    Because it has no memory, it cannot recognize that it has already tried
    'turn_left' in this exact spot before -- so in a corner or U-shaped wall
    it can get stuck cycling forever: turn_left -> still wall_ahead ->
    turn_left -> ... or bounce between 'move_forward' and 'turn_left'
    without ever escaping.
    """

    def sense_and_act(self, percept):
        # Strict condition-action (IF-THEN) rules, checked in priority order.
        if percept['food_here']:
            return 'suck'
        elif percept['wall_ahead']:
            return 'turn_left'
        else:
            return 'move_forward'


class ModelBasedAgent:
    """
    A model-based reflex agent. Unlike SimpleReflexAgent, it keeps an internal
    state that lets it track (via dead reckoning) where it believes it is and
    which way it believes it's facing, purely from knowing the effects of its
    own past actions -- it never sees global coordinates from the environment.

    Internal state:
      - self.pos:            believed (relative) position, starting at (0, 0)
      - self.facing:         believed facing direction, starting at 'Up'
      - self.visited_cells:  set of relative cells the agent believes it has
                              already visited
      - self.last_action:    the action taken on the previous call, used to
                              update the position/facing model this turn

    This lets the agent notice when its usual "hug the wall and turn_left"
    strategy would send it back into an already-visited cell, and pick a
    different action (turn_right) to break out of the loop.
    """

    def __init__(self):
        self.pos = (0, 0)
        self.facing = 'Up'
        self.visited_cells = {(0, 0)}
        self.last_action = None
        self.last_wall_ahead = False  # wall_ahead value at the time last_action was chosen
        # Maps (pos, facing) -> the turn we chose last time we were stuck here.
        # Lets us detect "I've already tried this rotation from this exact
        # spot and it didn't work" instead of re-deriving the same choice
        # from a single neighbor check every time (which caused turn_left
        # and turn_right to simply undo each other forever).
        self.stuck_turn_history = {}

    def _update_state(self, percept):
        """Transition model: apply the effect of the PREVIOUS action to update
        our belief about position/facing. We use self.last_wall_ahead (captured
        when the previous action was chosen) rather than the current percept,
        since a 'move_forward' only succeeds if there was no wall ahead THEN."""
        if self.last_action == 'move_forward' and not self.last_wall_ahead:
            dx, dy = DIRECTION_DELTA[self.facing]
            self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        elif self.last_action == 'turn_left':
            self.facing = TURN_LEFT_MAP[self.facing]
        elif self.last_action == 'turn_right':
            self.facing = TURN_RIGHT_MAP[self.facing]
        # 'suck', None, or a blocked move_forward -> no position/facing change

    def _relative_cell(self, turn=None):
        """Compute the (relative) cell in front of the agent, optionally as if
        it first turned left/right, without actually changing its state."""
        facing = self.facing
        if turn == 'left':
            facing = TURN_LEFT_MAP[facing]
        elif turn == 'right':
            facing = TURN_RIGHT_MAP[facing]
        dx, dy = DIRECTION_DELTA[facing]
        return (self.pos[0] + dx, self.pos[1] + dy)

    def sense_and_act(self, percept):
        # 1. Update internal state using the transition model (effect of last action)
        self._update_state(percept)
        self.visited_cells.add(self.pos)

        # 2. IF-THEN rules that now query memory instead of acting blindly
        if percept['food_here']:
            action = 'suck'

        elif percept['wall_ahead']:
            state = (self.pos, self.facing)

            if state in self.stuck_turn_history:
                # We've been stuck at this exact spot, facing this exact way,
                # before -- whatever we tried then didn't get us anywhere
                # (we're right back here). Force the OPPOSITE turn this time
                # instead of re-deriving the same choice, which would just
                # undo the last rotation and flip-flop forever.
                previous_turn = self.stuck_turn_history[state]
                action = 'turn_left' if previous_turn == 'turn_right' else 'turn_right'
            else:
                left_cell = self._relative_cell(turn='left')
                # IF wall_ahead AND left_is_visited THEN turn_right (avoid re-treading
                # the same old escape route -- try the other way instead)
                if left_cell in self.visited_cells:
                    action = 'turn_right'
                else:
                    action = 'turn_left'

            self.stuck_turn_history[state] = action

        else:
            ahead_cell = self._relative_cell()
            left_cell = self._relative_cell(turn='left')
            # IF the cell ahead is already visited but turning left leads
            # somewhere new, prefer exploring the unvisited direction instead
            # of retracing the same loop.
            if ahead_cell in self.visited_cells and left_cell not in self.visited_cells:
                action = 'turn_left'
            else:
                action = 'move_forward'

        # Remember what we did and whether a wall was ahead, so next call's
        # _update_state knows whether a 'move_forward' actually succeeded.
        self.last_wall_ahead = percept['wall_ahead']
        self.last_action = action
        return action


class GridGameGUI:
    """Tkinter wrapper that dynamically scales cell sizes to keep larger grids on screen."""

    def __init__(self, root, width=10, height=10, num_food=12, num_opponents=2, walls=None, agent_type='model_based'):
        self.root = root
        self.root.title("IT3012 - Scalable Multi-Agent Grid Hunt")

        self.env = VisualGridHuntGame(width=width, height=height, num_food=num_food, num_opponents=num_opponents,
                                      custom_walls=walls)

        if agent_type == 'simple_reflex':
            self.agent = SimpleReflexAgent()
        else:
            self.agent = ModelBasedAgent()

        max_canvas_dim = 600
        self.cell_size = max(20, min(max_canvas_dim // self.env.width, max_canvas_dim // self.env.height))

        canvas_w = self.env.width * self.cell_size
        canvas_h = self.env.height * self.cell_size

        self.canvas = tk.Canvas(root, width=canvas_w, height=canvas_h, bg="white")
        self.canvas.pack()

        self.label = tk.Label(root, text="Score: 0 | Steps: 0", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn = tk.Button(root, text="Start Simulation", command=self.run_loop, font=("Arial", 12), bg="#000066",
                             fg="white")
        self.btn.pack(pady=5)

        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")

        for x in range(self.env.width):
            for y in range(self.env.height):
                x1 = x * self.cell_size
                y1 = (self.env.height - 1 - y) * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                color = "#f1f5f9" if (x, y) not in self.env.walls else "#64748b"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#cbd5e1")

                if self.cell_size >= 40 and (x, y) in self.env.walls:
                    self.canvas.create_text(x1 + self.cell_size / 2, y1 + self.cell_size / 2,
                                            text="W", fill="white",
                                            font=("Arial", 8, "bold"))

        # Visualize the agent's memory (only meaningful for ModelBasedAgent).
        # Its visited_cells are relative to its own start (0,0)/'Up', which
        # matches the env's start, so they map 1:1 onto absolute grid cells.
        visited_cells = getattr(self.agent, 'visited_cells', None)
        if visited_cells:
            for vx, vy in visited_cells:
                if 0 <= vx < self.env.width and 0 <= vy < self.env.height:
                    x1 = vx * self.cell_size + 2
                    y1 = (self.env.height - 1 - vy) * self.cell_size + 2
                    x2 = x1 + self.cell_size - 4
                    y2 = y1 + self.cell_size - 4
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="#93c5fd", width=2)

        for fx, fy in self.env.food_positions:
            offset = self.cell_size * 0.25
            x1 = fx * self.cell_size + offset
            y1 = (self.env.height - 1 - fy) * self.cell_size + offset
            self.canvas.create_oval(x1, y1,
                                    x1 + self.cell_size * 0.5,
                                    y1 + self.cell_size * 0.5,
                                    fill="#f59e0b",
                                    outline="#d97706")

        for tx, ty in self.env.toxic_traps:
            offset = self.cell_size * 0.25
            x1 = tx * self.cell_size + offset
            y1 = (self.env.height - 1 - ty) * self.cell_size + offset
            self.canvas.create_oval(x1, y1,
                                    x1 + self.cell_size * 0.5,
                                    y1 + self.cell_size * 0.5,
                                    fill="purple"
             )

        for ox, oy in self.env.opponents:
            offset = self.cell_size * 0.2
            x1 = ox * self.cell_size + offset
            y1 = (self.env.height - 1 - oy) * self.cell_size + offset
            self.canvas.create_rectangle(
                x1, y1,
                x1 + self.cell_size * 0.6,
                y1 + self.cell_size * 0.6,
                fill="#990000",
                outline="#7a0000"
            )

        ax, ay = self.env.agent_pos
        offset = self.cell_size * 0.15
        x1 = ax * self.cell_size + offset
        y1 = (self.env.height - 1 - ay) * self.cell_size + offset
        x2 = x1 + self.cell_size * 0.7
        y2 = y1 + self.cell_size * 0.7
        self.canvas.create_oval(
            x1, y1, x2, y2,
            fill="#000066",
            outline="#1e3a8a"
        )

        # Draw a small line indicating which way the agent is facing
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = DIRECTION_DELTA[self.env.facing]
        fx2 = cx + dx * self.cell_size * 0.4
        fy2 = cy - dy * self.cell_size * 0.4  # canvas y is flipped vs grid y
        self.canvas.create_line(cx, cy, fx2, fy2, fill="yellow", width=3)

    def run_loop(self):
        self.btn.config(state="disabled")

        def step():
            if not self.env.is_done():
                percept = self.env.get_percept()
                action = self.agent.sense_and_act(percept)
                self.env.execute_action(action)

                self.draw_grid()
                self.label.config(
                    text=f"Score: {self.env.score} | Steps: {self.env.steps} | "
                         f"Action: {action} | Facing: {self.env.facing}"
                )
                self.root.after(250, step)
            else:
                end_text = f"Collision! Game Over! Final Score: {self.env.score}" if self.env.collision else f"Finished! Final Score: {self.env.score}"
                self.label.config(text=end_text)
                self.btn.config(state="normal")

        step()


if __name__ == "__main__":
    root = tk.Tk()
    app = GridGameGUI(root, width=12, height=12, num_food=15, num_opponents=0)
    root.mainloop()
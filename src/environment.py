"""Grid environment for the UAV surveillance optimization simulation."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Environment:
    size: int = 50
    num_drones: int = 3
    num_waypoints: int = 8
    seed: int = 7

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.no_fly = np.zeros((self.size, self.size), dtype=bool)
        self.hazard = np.zeros((self.size, self.size), dtype=float)
        self.priority = np.zeros((self.size, self.size), dtype=float)
        self.starts = [(3, 3), (3, self.size - 4), (self.size - 4, 3)]
        self._create_environment()
        self._free_cells = np.argwhere(~self.no_fly)

    def _create_environment(self):
        # Rectangular no-fly zones. Coordinates are stored as (x, y).
        blocks = [(15, 7, 20, 17), (28, 25, 36, 34), (8, 31, 14, 39)]
        for x1, y1, x2, y2 in blocks:
            self.no_fly[x1:x2 + 1, y1:y2 + 1] = True

        # Smooth-ish hazard pockets with values from 0 to 1.
        xx, yy = np.meshgrid(np.arange(self.size), np.arange(self.size), indexing="ij")
        centers = [(24, 12, 7, 0.9), (38, 39, 8, 0.8), (12, 23, 6, 0.7)]
        for cx, cy, radius, severity in centers:
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            self.hazard = np.maximum(
                self.hazard, severity * np.clip(1 - dist / radius, 0, 1)
            )
        self.hazard[self.no_fly] = 1.0

        # Priority targets: deterministic, reproducible and away from blocked cells.
        candidate_points = [(10, 8), (23, 6), (25, 20), (40, 14), (42, 28),
                            (20, 39), (33, 42), (7, 25), (44, 44), (17, 27),
                            (5, 43), (30, 10), (38, 20), (22, 32), (45, 6)]
        for idx, (x, y) in enumerate(candidate_points):
            if not self.no_fly[x, y]:
                self.priority[x, y] = 1.0 + (idx % 3) * 0.5

        # Ensure starts are open.
        for x, y in self.starts:
            self.no_fly[x, y] = False
            self.hazard[x, y] = 0.0

    def in_bounds(self, point):
        x, y = int(point[0]), int(point[1])
        return 0 <= x < self.size and 0 <= y < self.size

    def is_blocked(self, point):
        if not self.in_bounds(point):
            return True
        return bool(self.no_fly[int(point[0]), int(point[1])])

    def safe_point(self, point):
        """Clip a coordinate and, if blocked, move to nearest free cell."""
        x = int(np.clip(round(float(point[0])), 0, self.size - 1))
        y = int(np.clip(round(float(point[1])), 0, self.size - 1))
        if not self.no_fly[x, y]:
            return (x, y)
        free = self._free_cells
        idx = np.argmin((free[:, 0] - x) ** 2 + (free[:, 1] - y) ** 2)
        return tuple(map(int, free[idx]))

    def line_cells(self, a, b):
        """Return integer grid cells along a line (Bresenham-style sampling)."""
        x0, y0 = map(int, a)
        x1, y1 = map(int, b)
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        cells = []
        for t in np.linspace(0, 1, steps + 1):
            x = int(round(x0 + t * (x1 - x0)))
            y = int(round(y0 + t * (y1 - y0)))
            cell = (x, y)
            if not cells or cells[-1] != cell:
                cells.append(cell)
        return cells

    def repair_route(self, start, waypoints):
        """Repair waypoints and conservatively remove segments crossing no-fly cells.

        If a straight segment crosses a blocked cell, the waypoint is replaced by
        the previous safe point. This is fast and guarantees the represented
        segment does not cross the obstacle, although it may reduce coverage.
        """
        route = [tuple(map(int, start))]
        for raw in waypoints:
            target = self.safe_point(raw)
            previous = route[-1]
            cells = self.line_cells(previous, target)
            if any(self.is_blocked(c) for c in cells[1:]):
                target = previous
            route.append(target)
        return route

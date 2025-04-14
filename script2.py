from __future__ import annotations
import copy
import hashlib
from collections import deque
import random
import heapq
import time

class Grid:
    def __init__(self, board: list[list[int]]):
        self.board = copy.deepcopy(board)
        self.parent = None
        self.depth = 0
        hash(self)
    
    def set_row_col(self, row: int, col: int):
        self.row = row
        self.col = col
        self.board[row][col] = 2
    
    def __stringify_grid(grid) -> str:
        chars = " ~#"
        return "\n".join(["".join([f"[{chars[c]}]" for c in row]) for row in grid])

    def __str__(self) -> str:
        return Grid.__stringify_grid(self.board)
    
    def __hash__(self) -> int:
        s = "".join(["".join([f"{c}" for c in row]) for row in self.board])
        # Vérifie si row et col sont définis, sinon utilise des valeurs par défaut
        row_str = str(self.row) if hasattr(self, 'row') else "-1"
        col_str = str(self.col) if hasattr(self, 'col') else "-1"
        s += row_str + "," + col_str  # Ligne demandée par le professeur
        self.__hash_code = int.from_bytes(hashlib.md5(s.encode()).digest(), "little")
        return self.__hash_code
    
    def __eq__(self, value: 'Grid') -> bool:
        return self.__hash_code == value.__hash_code
    
    def clone(self, row: int, col: int) -> 'Grid':
        v = Grid(self.board)
        v.set_row_col(row, col)
        v.parent = self
        v.depth = self.depth + 1
        hash(v)
        return v
    
    def is_goal(self) -> bool:
        for row in self.board:
            if 1 in row:
                return False
        return True
    
    def actions(self) -> list['Grid']:
        tries = []
        if self.row > 0 and self.board[self.row-1][self.col] == 1:
            tries.append((self.row-1, self.col))
        if self.row < len(self.board)-1 and self.board[self.row+1][self.col] == 1:
            tries.append((self.row+1, self.col))
        if self.col > 0 and self.board[self.row][self.col-1] == 1:
            tries.append((self.row, self.col-1))
        if self.col < len(self.board[0])-1 and self.board[self.row][self.col+1] == 1:
            tries.append((self.row, self.col+1))
        return [self.clone(row1, col1) for row1, col1 in tries]
    
    def extract_submatrix(self, k: int) -> list[list[int]]:
        """Extrait une sous-matrice kxk centrée sur (row, col)."""
        half_k = k // 2
        r_start = max(0, self.row - half_k)
        r_end = min(len(self.board), self.row + half_k + 1)
        c_start = max(0, self.col - half_k)
        c_end = min(len(self.board[0]), self.col + half_k + 1)
        sub = [row[c_start:c_end] for row in self.board[r_start:r_end]]
        # Remplir avec des 0 si hors limites
        result = [[0] * k for _ in range(k)]
        for i in range(len(sub)):
            for j in range(len(sub[i])):
                result[i + (k - len(sub)) // 2][j + (k - len(sub[i])) // 2] = sub[i][j]
        return result
    
    def calc_pk(self, k: int) -> int:
        """Calcule p_k avec solve_rec."""
        submatrix = self.extract_submatrix(k)
        n = sum(row.count(1) for row in submatrix)
        covered = solve_rec(submatrix, k // 2, k // 2, 0, n)
        return n - covered
    
    def solve_breadth(self, max_nodes=5000000):
        queue = deque([(self, [(self.row, self.col)])])
        visited = {hash(self)}
        nodes = 0
        while queue:
            if nodes >= max_nodes:
                print(f"Limite de {max_nodes} nœuds atteinte dans Breadth")
                return None
            grid, chemin = queue.popleft()
            if grid.is_goal():
                print(f"Solution trouvée après {nodes} nœuds dans Breadth")
                return chemin
            for next_grid in grid.actions():
                if hash(next_grid) not in visited:
                    visited.add(hash(next_grid))
                    new_chemin = chemin + [(next_grid.row, next_grid.col)]
                    queue.append((next_grid, new_chemin))
                    nodes += 1
                    if nodes % 10000 == 0:
                        print(f"Exploration Breadth: {nodes} nœuds")
        print(f"Épuisé après {nodes} nœuds dans Breadth")
        return None
    
    def solve_depth(self, max_nodes=500000):
        visited = {hash(self)}
        stack = [(self, [(self.row, self.col)])]
        nodes = 0
        while stack:
            if nodes >= max_nodes:
                print(f"Limite de {max_nodes} nœuds atteinte dans Depth")
                return None
            grid, chemin = stack.pop()
            if grid.is_goal():
                print(f"Solution trouvée après {nodes} nœuds dans Depth")
                return chemin
            actions = grid.actions()
            if not actions:
                continue
            for next_grid in actions:
                if hash(next_grid) not in visited:
                    visited.add(hash(next_grid))
                    new_chemin = chemin + [(next_grid.row, next_grid.col)]
                    stack.append((next_grid, new_chemin))
                    nodes += 1
                    if nodes % 10000 == 0:
                        print(f"Exploration Depth: {nodes} nœuds")
        print(f"Épuisé après {nodes} nœuds dans Depth")
        return None
    
    def solve_random(self):
        visited = {hash(self)}
        chemin = [(self.row, self.col)]
        current = self
        while True:
            if current.is_goal():
                return chemin
            actions = current.actions()
            possible = [g for g in actions if hash(g) not in visited]
            if not possible:
                return None
            next_grid = random.choice(possible)
            visited.add(hash(next_grid))
            chemin.append((next_grid.row, next_grid.col))
            current = next_grid
    
    def solve_heur(self, a=1, b=1, c=10, max_nodes=1000000):
        counter = 0
        heap = [(0, 0, counter, self, [(self.row, self.col)])]
        visited = {hash(self)}
        nodes = 0
        while heap:
            if nodes >= max_nodes:
                print(f"Limite de {max_nodes} nœuds atteinte dans Heur ({a},{b},{c})")
                return None
            f, d, _, grid, chemin = heapq.heappop(heap)
            if grid.is_goal():
                print(f"Solution trouvée après {nodes} nœuds dans Heur ({a},{b},{c})")
                return chemin
            for next_grid in grid.actions():
                if hash(next_grid) not in visited:
                    visited.add(hash(next_grid))
                    p5 = next_grid.calc_pk(5)
                    p3 = next_grid.calc_pk(3)
                    h = a * p5 + b * p3 - (d + 1) // c
                    new_f = d + 1 + h  # f = g + h
                    counter += 1
                    new_chemin = chemin + [(next_grid.row, next_grid.col)]
                    heapq.heappush(heap, (new_f, d + 1, counter, next_grid, new_chemin))
                    nodes += 1
                    if nodes % 10000 == 0:
                        print(f"Exploration Heur ({a},{b},{c}): {nodes} nœuds")
        print(f"Épuisé après {nodes} nœuds dans Heur ({a},{b},{c})")
        return None

def solve_rec(world, row: int, col: int, nb: int, n: int) -> int:
    if nb == n:
        return n
    tries = []
    if row > 0 and world[row-1][col] == 1:
        tries.append((row-1, col))
    if row < len(world)-1 and world[row+1][col] == 1:
        tries.append((row+1, col))
    if col > 0 and world[row][col-1] == 1:
        tries.append((row, col-1))
    if col < len(world[0])-1 and world[row][col+1] == 1:
        tries.append((row, col+1))
    mx = nb
    for row1, col1 in tries:
        world[row1][col1] = 2
        sol = solve_rec(world, row1, col1, nb + 1, n)
        world[row1][col1] = 1
        if mx < sol:
            mx = sol
        if mx == n:
            return n
    return mx


benchmark1 = [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
benchmark2 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]]
benchmark3 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
benchmark4 = [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0], [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
benchmark5 = [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0], [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]
benchmark7 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]
benchmark8 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

def find_first_blue(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 1:
                return i, j
    return 3, 3  # Par défaut

if __name__ == "__main__":
   
    benchmarks = [benchmark1, benchmark2, benchmark3, benchmark4, benchmark5, benchmark7, benchmark8]
    methods = [
        ("Depth", lambda g: g.solve_depth(max_nodes=500000)),
        ("Breadth", lambda g: g.solve_breadth(max_nodes=5000000)),
        ("Random", lambda g: g.solve_random()),
        ("Heur (1,1,10)", lambda g: g.solve_heur(1, 1, 10, max_nodes=1000000)),
        ("Heur (0,1,50)", lambda g: g.solve_heur(0, 1, 50, max_nodes=1000000)),
    ]
    
    for i, bench in enumerate(benchmarks, 1):
        print(f"\nBenchmark {i}:")
        gr = Grid(bench)
        row, col = find_first_blue(bench)
        gr.set_row_col(row, col)
        print(f"Position initiale : ({row}, {col})")
        for name, method in methods:
            print(f"Début de {name}...")
            start = time.time()
            result = method(gr)
            end = time.time()
            if result:
                print(f"{name}: Temps = {end - start:.3f}s, Chemin = {result}")
            else:
                print(f"{name}: Temps = {end - start:.3f}s, Chemin = None")
            print(f"Fin de {name}")
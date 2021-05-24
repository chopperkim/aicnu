"""
    A* search practice code

    This code is borrowed from https://www.youtube.com/watch?v=JtiK0DOeI4A
    and annotated for education purpose
    
"""

import pygame
import math
from queue import PriorityQueue

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

## -- color define part -- ##
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)
# ----------------------------

class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE  # <-- default color is white
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE  # <-- set ORANGE color for start spot(cell)

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        ## in this function, 
        ## add neighbors spots in a grid to a spot.
        ## Note that
        ## --> the neighborhood is corresponding to a edge between two nodes in A* search
        ## --> if the neighbor spot is barrier?? --> do not construct edge 
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False


# heuristic function
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()


def algorithm(draw, grid, start, end):
    # note that draw is a function

    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}

    # A* search : f(n) = g(n) + h(n) 
    #   - g(n) : actual cost from START to n spot
    #   - h(n) : guessed cost from n to END spot  <-- calculated by heuristic function
    #          : the heuristic function is quite simple 
    #          : = abs(x1 - x2) + abs(y1 - y2)
    #          # = it does not care 'barrier' spots at all.

    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score

                # search cost function
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos()) # actual traverse cost + guess cost

                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False


def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            ## create spot(cells)  
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)

    return grid


def draw_grid(win, rows, width):
    ## in this function, actually draw spots in a graphic environment
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col


def main(win, width):
  
    # draw grid - environment
    ROWS = 50
    grid = make_grid(ROWS, width)

    start = None
    end = None

    run = True
    ## -- main event loop -- ##
    while run:
        # draw or refresh spots 
        draw(win, grid, ROWS, width)

        # wait event from user or internal behaviour
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # EVENT - mouse left click
            if pygame.mouse.get_pressed()[0]: 
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]

                # at first mouse click --- set start spot 
                if not start and spot != end:
                    start = spot  # <-- the firstly clikced spot become a start spot
                    start.make_start()

                # at second mouse click --- set goal(end) spot
                elif not end and spot != start:
                    end = spot    # <-- the secondly clikced spot become a end(goal) spot
                    end.make_end()

                # at other clicks? --> make the clicked cells as barrier cells
                elif spot != end and spot != start:
                    spot.make_barrier()

            # EVENT - mouse right click 
            # --> reset the spot's roll
            # ( you don't need to understand the detail logic for A* )
            elif pygame.mouse.get_pressed()[2]: # RIGHT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            # EVENT - keyboard 
            if event.type == pygame.KEYDOWN:

                # at SPACE-BAR keyboard typing, triger to algorithm to find the path
                if event.key == pygame.K_SPACE and start and end:

                    ## for A* search, construct edge-connection information
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)

                    algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)

                ## at c, clear all
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

    pygame.quit()

main(WIN, WIDTH)
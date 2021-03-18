import matplotlib.pyplot as plt
import heapq

import copy
from time import sleep
from random import randint
from itertools import permutations

def line(ps):
    xc=[]
    yc=[]
    for i in (range(0,len(ps))):
        x=ps[i][0]
        y=ps[i][1]
        xc.append(x)
        yc.append(y)
    return xc,yc


def distance(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])
    # return ((p[0] - q[0])**2 + (p[1] - q[1])**2)**0.5

class Maze:
    def __init__(self, n, m, t, s, start, weapon_power=2):
        
        self.n = n
        self.m = m
        self.t = t
        self.s = s
        self.weapon_power = weapon_power
        self.start = start
        self.coins = []

    def make_board(self):
        self.b = [[0 for _ in range(self.n)] for _ in range(self.n)]
        rock, wood, coin = self.t, self.m, self.s
        while wood > 0:
            x = randint(0, self.n-1)
            y = randint(0, self.n-1)
            if self.b[x][y] != 0 or (x, y) == self.start:
                continue
            self.b[x][y] = 2
            wood -= 1
        while rock > 0:
            x = randint(0, self.n-1)
            y = randint(0, self.n-1)
            if self.b[x][y] != 0 or (x, y) == self.start:
                continue
            self.b[x][y] = 5
            rock -= 1
        while coin > 0:
            x = randint(0, self.n-1)
            y = randint(0, self.n-1)
            if self.b[x][y] != 0 or (x, y) in self.coins + [self.start]:
                continue
            self.coins.append((x, y))
            coin -= 1


    def run(self):
        fig,ax=plt.subplots()
        ax.clear()
        ax.scatter(self.start[1],self.start[0])
        for e in self.coins:
            ax.scatter(e[1],e[0], c=['yellow'])
        ax.imshow(self.b,cmap=plt.cm.Spectral)
        

        for p, p_type in self.a_star():
            if p_type == 'points':
                ax.plot([p[0][1], p[1][1]], [p[0][0], p[1][0]], color="red")
                plt.pause(0.00005)
            if p_type == 'subpath':
                xs, ys = line(p)
                ax.plot(ys, xs, color='white')
                plt.pause(0.005)
            if p_type == 'mainpath':
                p, c = p
                plt.title(f'{c}')
                for i, point in enumerate(p[:-1]):
                    ax.plot([point[1], p[i+1][1]], [point[0], p[i+1][0]], color="black")
                    plt.pause(0.03)
        
        plt.show()

    def a_star(self):
        tc = 0
        self.board = copy.deepcopy(self.b)# self.b.copy()
        ends = self.coins.copy()
        start = self.start
        cost = lambda a, b: distance(a, b) + self.board[b[0]][b[1]]
        heuristic = lambda p: min([distance(p, e) for e in ends])
        childs_maker = lambda p: [(p[0]+i, p[1]+j) for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)] if (-1<p[0]+i<self.n and -1<p[1]+j<self.n)]
        path = []
        parents = {}
        g_s = {start: 0}
        f_s = {start: heuristic(start)}
        unchecked_points = [(f_s[start], start)]  # [(f, point), ...]
        checked_points = set()
        while unchecked_points and ends:
            current_point = heapq.heappop(unchecked_points)[1]
            checked_points.add(current_point)

            if current_point in ends:
                e = current_point
                ends.remove(e)
                p = []
                c = 0
                while e in parents:
                    c += 1 + self.board[e[0]][e[1]]
                    p.append(e)
                    self.board[e[0]][e[1]] = 0
                    e = parents[e]
                p.append(e)
                p = p[::-1]
                path += p
                tc += c
                yield p, 'subpath'
                if not ends:
                    yield [path, tc], 'mainpath'
                    return 0

                ######
                checked_points = set()
                g_s = {current_point: 0}
                f_s = {current_point: heuristic(current_point)}
                parents = {}
                unchecked_points = [(f_s[current_point], current_point)]
                continue

            for child in childs_maker(current_point):
                if self.board[child[0]][child[1]] > self.weapon_power:
                    continue
                g = g_s[current_point] + cost(current_point, child)

                if child in checked_points and child in g_s:
                    if g >= g_s[child]:
                        continue
                
                if child not in [up[1] for up in unchecked_points] or g < g_s[child]:
                    yield [current_point, child], 'points'
                    
                    parents[child] = current_point
                    f = g + heuristic(current_point)
                    g_s[child] = g
                    f_s[child] = f
                    heapq.heappush(unchecked_points, (f, child))
        return False




n = int(input('enter n = '))
m = int(input('enter m = '))
t = int(input('enter t = '))
s = int(input('enter s = '))
wp = int(input('enter weapon power(2 or 5) = '))
start_x = int(input('enter x of start position = '))
start_y = int(input('enter y of start position = '))

g = Maze(n=n, m=m, t=t, s=s, start=(start_y, start_x), weapon_power=wp)
g.make_board()
g.run()

import matplotlib.pyplot as plt
import heapq

import copy
from time import sleep
from random import randint
from itertools import permutations

import numpy as np
from itertools import combinations
import itertools



def line(route):
    xc=[]
    yc=[]
    for i in (range(0,len(route))):
        x=route[i][0]
        y=route[i][1]
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


    
    def solve_tsp_dynamic(self, G):
        #calc all lengths
        all_distances = G #[[length(x,y) for y in points] for x in points]
        #initial value - just distance from 0 to every other point + keep the track of edges
        A = {(frozenset([0, idx+1]), idx+1): (dist, [0,idx+1]) for idx,dist in enumerate(all_distances[0][1:])}
        cnt = len(G)
        for m in range(2, cnt):
            B = {}
            for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
                for j in S - {0}:
                    
                    B[(S, j)] = min( [(A[(S-{j},k)][0] + all_distances[k][j], A[(S-{j},k)][1] + [j]) for k in S if k != 0 and k!=j])  #this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
            print(B)
            A = B
            
        #res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
        res = min([(A[d][0], A[d][1]) for d in iter(A)])
        
        return res

    def run(self):
        fig,ax=plt.subplots()
        ax.scatter(self.start[1],self.start[0])
        for e in self.coins:
            ax.scatter(e[1],e[0], c=['yellow'])
        ax.imshow(self.b,cmap=plt.cm.Spectral)
        plt.show()


        vs = [self.start] + self.coins
        d_vs = {n: v for n, v in enumerate(vs)}
        
        n = len(vs)
        G = np.zeros((n, n))
        P = [[[] for _ in range(n)] for __ in range(n)]
        for i, r in enumerate(G):
            #self.board = copy.deepcopy(self.b)# self.b.copy()
        
            for j, c in enumerate(r):
                if G[j][i] != 0:
                    G[i][j] = G[j][i]
                    P[i][j] = P[j][i]
                    continue
                a = self.a_star(d_vs[i], d_vs[j])
                
                G[i][j], P[i][j] = a[0], a[1]

        total_cost, paths = self.solve_tsp_dynamic(G)

        fig,ax=plt.subplots()
        ax.imshow(self.b,cmap=plt.cm.Spectral)
        ax.scatter(self.start[1],self.start[0])
        for e in self.coins:
            ax.scatter(e[1],e[0], c=['yellow'])
        
        plt.title(str(total_cost))
        for i_p, p in enumerate(paths[:-1]):
            
            ps = P[p][paths[i_p+1]]
            if d_vs[p] != ps[0]:
                ps = ps[::-1]
            for i, point in enumerate(ps[:-1]):
                ax.plot([point[1], ps[i+1][1]], [point[0], ps[i+1][0]], color="green")
                plt.pause(0.7)
        plt.show()


        
    def a_star(self, start, end):
        board = copy.deepcopy(self.b)
        
        cost = lambda a, b: distance(a, b) + board[b[0]][b[1]]
        heuristic = lambda p: distance(p, end) #min([distance(p, e) for e in ends])
        childs_maker = lambda p: [(p[0]+i, p[1]+j) for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)] if (-1<p[0]+i<self.n and -1<p[1]+j<self.n)]
        path = []
        parents = {}
        g_s = {start: 0}
        f_s = {start: heuristic(start)}
        unchecked_points = [(f_s[start], start)]  # [(f, point), ...]
        checked_points = set()
        while unchecked_points:
            current_point = heapq.heappop(unchecked_points)[1]
            checked_points.add(current_point)

            if current_point == end:
                e = current_point
                
                p = []
                c = 0
                while e in parents:
                    c += 1 + board[e[0]][e[1]]
                    p.append(e)
                    board[e[0]][e[1]] = 0
                    e = parents[e]
                p.append(e)
                p = p[::-1]
                path += p
                #yield p, 'subpath'
                return c, p
                

            for child in childs_maker(current_point):
                if board[child[0]][child[1]] > self.weapon_power:
                    continue
                g = g_s[current_point] + cost(current_point, child)

                if child in checked_points and child in g_s:
                    if g >= g_s[child]:
                        continue
                
                if child not in [up[1] for up in unchecked_points] or g < g_s[child]:
                    #yield [current_point, child], 'points'
                    
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

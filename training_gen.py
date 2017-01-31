# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 23:53:00 2017

@author: brian
"""
import numpy as np
from action import Action
from action import Direction
from action import Symbol

def input_gen():
    length = int(np.random.geometric(0.1))
    string = ""
    for i in range(length):
        if np.random.random() < 0.5:
            string += "0"
        else:
            string += "1"
    return string

def target_actions_reverse(s):
    acts = []
    for i in range(len(s) + 1):
        act = Action(Symbol.empty_symbol(), Direction("right"))
        acts.append(act)
    acts.append(Action(Symbol.empty_symbol(), Direction("left")))
    for i in range(len(s)):
        act = Action(Symbol(s[len(s) - 1 - i]), Direction("left"))
        acts.append(act)
    acts.append(Action(Symbol.term_symbol(), Direction("left")))
    return acts
    
def copy_input_gen():
    length = int(np.random.geometric(0.1))
    string = ""
    for i in range(length):
        if np.random.random() < 0.5:
            string += "0"
        else:
            string += "1"
    return string
        
def copy_target_actions(s):
    acts = []
    acts.append(Action(Symbol.empty_symbol(), Direction("right")))
    for i in range(len(s)):
        act = Action(Symbol(s[i]), Direction("right"))
        acts.append(act)
    acts.append(Action(Symbol.term_symbol(), Direction("left")))
    return acts
    
def walk_gen():
    l = np.random.randint(2, 7)
    w = np.random.randint(2, 7)
    p = 0.10
    ret = [['e' for j in range(l)] for i in range(w)]
    acts = []
    loc = [0,0]
    start_loc = [0, np.random.randint(0, l)]
    while(loc[1] < start_loc[1]):
        sym = Symbol.empty_symbol()
        direc = Direction('right')
        acts.append(Action(sym, direc))
        loc[1] += 1
    while(True):
        if (np.random.random() < p):
            break
        mov = []
        if loc[0] > 0 and ret[loc[0] - 1][loc[1]] == 'e':
            mov += ['u']
        if loc[1] > 0 and ret[loc[0]][loc[1] - 1] == 'e':
            mov += ['l']
        if loc[0] < w-1 and ret[loc[0] + 1][loc[1]] == 'e':
            mov += ['d']
        if loc[1] < l-1 and ret[loc[0]][loc[1] + 1] == 'e':
            mov += ['r']
        if len(mov) == 0:
            break
        mov = np.random.choice(mov)
        sym = Symbol(mov)
        ret[loc[0]][loc[1]] = mov
        direc = None
        if mov == 'u':
            loc[0] -= 1
            direc = Direction('up')
        if mov == 'l':
            loc[1] -= 1
            direc = Direction('left')
        if mov == 'd':
            loc[0] += 1
            direc = Direction('down')
        if mov == 'r':
            loc[1] += 1
            direc = Direction('right')
        act = Action(sym, direc)
        acts.append(act)
    act = Action(Symbol.term_symbol(), Direction('up'))
    acts.append(act)
    return ret, acts

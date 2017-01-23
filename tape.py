# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 20:27:44 2017

@author: brian
"""

from action import Symbol
from action import Direction
from action import Action

class InputTape1D():
    
    # string is the input string
    def __init__(self, string):
        self.rep = string
        self.symbols = []
        self.symbols.append(Symbol.start_symbol())
        for char in string:
            self.symbols.append(Symbol(char))
        self.symbols.append(Symbol.end_symbol())
        self.length = len(self.symbols)
        
    def __str__(self):
        return self.rep
        
    def get(self, loc):
        return self.symbols[loc]

class OutputTape1D():
    
    def __init__(self):
        self.symbols = []
        self.rep = ""
        
    def __str__(self):
        return self.rep
        
class InputTape2D():
    
    def __init__(self, arr):
        self.rep = ""
        self.symbols = [[Symbol('e') for j in range(len(arr[0]) + 2)] for i in range(len(arr) + 2)]
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                self.symbols[i+1][j+1] = Symbol(arr[i][j])
                self.rep += arr[i][j]
            self.rep += "\n"
        self.l = len(arr[0]) + 2
        self.w = len(arr) + 2
    
    def __str__(self):
        pass
    
    def get(self, loc):
        return self.symbols[loc[0]][loc[1]]
        
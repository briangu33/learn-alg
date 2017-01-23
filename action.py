# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 20:03:32 2017

@author: brian
"""

class Symbol():
    
    def __init__(self, rep, is_term = False):
        self.rep = rep
        self.is_start = (rep == 's')
        self.is_end = (rep == 'e')
        self.is_empty = (rep == '')
        self.is_term = is_term
        # a terminal symbol (one with is_term == True) has an empty
        # representation; the is_term property indicates that no more symbols
        # will be printed following it
        
    def __eq__(self, other):
        return (self.rep, self.is_term) == (other.rep, other.is_term)
        
    def __hash__(self):
        return hash((self.rep, self.is_term))
        
    def __str__(self):
        return self.rep
    
    @staticmethod
    def start_symbol():
        return Symbol('s')
    
    @staticmethod
    def end_symbol():
        return Symbol('e')
        
    @staticmethod
    def empty_symbol():
        return Symbol('')
        
    @staticmethod
    def term_symbol():
        return Symbol('', True)
        
class Direction():
    
    def __init__(self, name):
        self.name = name
        
    def __eq__(self, other):
        return self.name == other.name
        
    def __hash__(self):
        return hash(self.name)
        
    def __str__(self):
        return self.name

class Action():
    
    def __init__(self, symbol, direction):
        self.symbol = symbol
        self.direction = direction
    
    def __eq__(self, other):
        return (self.symbol, self.direction) == (other.symbol, other.direction)
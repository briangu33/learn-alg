# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 21:03:24 2017

@author: brian
"""

from action import Symbol
from action import Direction
from action import Action
from tape import InputTape1D
from tape import OutputTape1D
from transition import Transition

import numpy as np
import math

import sys

# if we end up needing to store more state-specific information, then we'll
# probably want to make state objects. for now, we'll keep all the machinery
# in PTMControl

def learn_rate(t, k=0.75, t0 = 100):
    return 10 * math.pow(t0 + t, -1*k)
    
def constant_learn_rate(t):
    return 0.5
    
class PTMControl():
    
    def __init__(self, n_states, alphabet, valid_dirs, learn_sched = constant_learn_rate):
        # by convention, an alphabet is a list of all possible symbols except
        # the empty symbol and the terminate symbol--i.e., valid input symbols
        
        # probability of being at state 0 is initially 1; all other
        # probabilities are initialized to 0
        self.timestep = 0    
        self.learn_sched = learn_sched
        
        self.n_states = n_states        
        
        self.p = [1. if i == 0 else 0. for i in range(self.n_states)]
        self.p_succeeding = 1.
        
        self.input_alphabet = [symbol for symbol in alphabet]
        self.output_alphabet = [symbol for symbol in alphabet \
            if not(symbol.is_start or symbol.is_end)]
        self.output_alphabet.append(Symbol.empty_symbol())
        self.output_alphabet.append(Symbol.term_symbol())
        
        self.valid_dirs = valid_dirs
                                   
        self.w = [dict() for i in range(n_states)]
        for i in range(n_states):
            self.w[i] = dict()
            for s in self.input_alphabet:
                self.w[i][s] = Transition(i, s, n_states, self.output_alphabet, self.valid_dirs)
                                                 
    def trans_sup(self, in_symbol, act):
        # a supervised transition
        p_next = [0. for i in range(self.n_states)]
        for from_state in range(self.n_states):
            trans = self.w[from_state][in_symbol]
            p_next += self.p[from_state] * trans.probs(act)
        self.p_succeeding *= self.p_success(in_symbol, act)
        self.p = p_next
        
    def p_success(self, in_symbol, act):
        p_success = 0.
        for from_state in range(self.n_states):
            trans = self.w[from_state][in_symbol]
            p_success += self.p[from_state] * trans.prob_of_action(act)
        return p_success
    
    def transition_and_update(self, in_symbol, act, learn_coeff):
        succ = self.p_success(in_symbol, act)
        """
        for from_state in range(self.n_states):
            new_trans = Transition(from_state, in_symbol, self.n_states, self.output_alphabet, self.valid_dirs)
            trans = self.w[from_state][in_symbol]
            multiplier = (1 - learn_coeff / succ * self.p[from_state] * trans.prob_of_action(act) / trans.total)
            if np.random.random() < 0.02:
                print learn_coeff, succ, trans.prob_of_action(act), trans.total
            new_trans.w_ind = trans.w_ind * multiplier
            for to_state in range(self.n_states):
                update_val = trans.get_w(to_state, act) \
                + learn_coeff * trans.prob_of_trans(to_state, act) * self.p[from_state] / succ # self.p_succeeding #  # 
                new_trans.upd_w(to_state, act, update_val)
            new_trans.renormalize()
            self.w[from_state][in_symbol] = new_trans
        """
        # a supervised transition which also updates weights
        for from_state in range(self.n_states):
            trans = self.w[from_state][in_symbol]
            new_trans = trans.deepcopy()
            for to_state in range(self.n_states):
                update_val = trans.get_w(to_state, act) \
                + learn_coeff * trans.prob_of_trans(to_state, act) * self.p[from_state] / succ # self.p_succeeding #  # 
                if np.isnan(update_val):
                    print trans.get_w(to_state, act), trans.prob_of_trans(to_state, act)
                    sys.exit()
                new_trans.upd_w(to_state, act, update_val)
            # new_trans.renormalize()
            self.w[from_state][in_symbol] = new_trans
        """        
        for from_state in range(self.n_states):
            trans = self.w[from_state][in_symbol]
            new_trans = trans.deepcopy()
            multiplier = learn_coeff / succ * self.p[from_state] * trans.prob_of_action(act)
            for to_state in range(self.n_states):
                for s in self.output_alphabet:
                    for d in self.valid_dirs:
                        this_act = Action(s, d)
                        update_val = trans.get_w(to_state, this_act) \
                        - multiplier * trans.prob_of_trans(to_state, this_act)
                        if np.isnan(update_val):
                            print trans.get_w(to_state, this_act), multiplier, trans.prob_of_trans(to_state, this_act)
                            sys.exit()
                        new_trans.upd_w(to_state, this_act, update_val)
            # new_trans.renormalize()
            self.w[from_state][in_symbol] = new_trans
        """
        """
        if b:
            
            for i in range(self.n_states):
                for s in self.input_alphabet:
                    if not self.w[i][s].locked:
                        self.w[i][s] = Transition(i, s, self.n_states, self.output_alphabet, self.valid_dirs)
                        
            for q in range(self.n_states):
                for s1 in self.input_alphabet:
                    j, a = self.w[q][s1].argmax()
                    s2 = a.symbol
                    d = a.direction
                    p = self.w[q][s1].prob_of_trans(j, a)
                    print q, "|", s1, "|", j, "|", s2.rep if not(s2.is_term) else "t", "|", d.name, "|", p, self.w[q][s1].locked
                    """
                    
        self.trans_sup(in_symbol, act)
        self.timestep += 1
    
    def run(self, input_tape, acts, train = False, verbose = False):
        if verbose:
            print "Input:", input_tape.rep
        self.p = [1. if i == 0 else 0. for i in range(self.n_states)]
        self.p_succeeding = 1.
        tape_loc = 0
        learned = None
        for act in acts:
            if train:
                temp = self.transition_and_update(input_tape.get(tape_loc), act, self.learn_sched(self.timestep))
                if not(temp is None):
                    learned = temp
            else:
                self.trans_sup(input_tape.get(tape_loc), act)
            if act.direction.name == 'left':
                tape_loc = max(0, tape_loc - 1)
            elif act.direction.name == 'right':
                tape_loc = min(input_tape.length - 1, tape_loc + 1)
            if verbose:
                print "Tape location:", tape_loc
                print self.p_succeeding, act.symbol.rep, act.direction.name, self.p
        self.p = [1. if i == 0 else 0. for i in range(self.n_states)]
        ret = self.p_succeeding
        self.p_succeeding = 1.
        """
        if train and np.random.random() < 0.1:
            for i in range(self.n_states):
                for s in self.input_alphabet:
                    self.w[i][s].renormalize()    
        """
        return ret    

    def run2D(self, input_tape, acts, train = False):
        self.p = [1. if i == 0 else 0. for i in range(self.n_states)]
        self.p_succeeding = 1.
        tape_loc = [1,1]
        for act in acts:
            if train:
                temp = self.transition_and_update(input_tape.get(tape_loc), act, self.learn_sched(self.timestep))
            else:
                self.trans_sup(input_tape.get(tape_loc), act)
            if act.direction.name == 'left':
                tape_loc[1] -= 1
            elif act.direction.name == 'right':
                tape_loc[1] += 1
            elif act.direction.name == 'up':
                tape_loc[0] -= 1
            else:
                tape_loc[0] += 1
        self.p = [1. if i == 0 else 0. for i in range(self.n_states)]
        ret = self.p_succeeding
        self.p_succeeding = 1.
        
        if train and np.random.random() < 0.01:
            for i in range(self.n_states):
                for s in self.input_alphabet:
                    self.w[i][s].renormalize()    
        
        return ret                    
        
"""
s1 = Symbol('s')
s2 = Symbol('0')
s3 = Symbol('1')
s4 = Symbol('e')

input_tape = InputTape1D('01')

a1 = Action(Symbol.empty_symbol(), Direction('right'))
a2 = Action(Symbol.empty_symbol(), Direction('right'))
a3 = Action(Symbol.empty_symbol(), Direction('right'))
a4 = Action(Symbol.empty_symbol(), Direction('left'))
a5 = Action(s3, Direction('left'))
a6 = Action(s2, Direction('left'))
a7 = Action(Symbol.term_symbol(), Direction('left'))

acts = [a1, a2, a3, a4, a5, a6, a7]
alphabet = [s1, s2, s3, s4]
valid_dirs = [Direction('left'), Direction('right')]
n_states = 3

ptmctrl = PTMControl(n_states, alphabet, valid_dirs)

# ptmctrl.train(input_tape, acts)
"""


        
        
        
        
        
        
        
        
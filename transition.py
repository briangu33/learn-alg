# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 17:51:23 2017

@author: brian
"""

from action import Action
from action import Direction
from action import Symbol

import numpy as np
import sys
import time
    
class Transition():
    # model of a transition from (state, read symbol) -> (state, write symbol,
    # direction)
    # provides fast methods for probabilities and conditioning on actions
    # as well as numerical stability for weight updates

    def __init__(self, state, read_sym, n_states, output_alph, valid_dirs):
        self.locked = False
        self.min_w = -1.
        self.max_w = 1.
        
        self.state = state
        self.read_sym = read_sym
        self.n_states = n_states
        
        self.output_alph = output_alph
        self.valid_dirs = valid_dirs
        self.sym_ind_map = dict()
        for i in range(len(output_alph)):
            self.sym_ind_map[output_alph[i]] = i
        self.dir_ind_map = dict()
        for i in range(len(valid_dirs)):
            self.dir_ind_map[valid_dirs[i]] = i

        self.w_ind = np.random.normal(scale = 0.2, size=(n_states,len(output_alph),len(valid_dirs)))
                    
        self.total = sum(sum(sum(np.exp(self.w_ind))))
        self.update_action_totals()
        
        # this is not necessarily good unless renormalize() was just called
        # in the future we'll maybe store a heap of (state, action) pairs
        # ordered by likelihood, so that this can always be right
        self.MLtrans = self.argmax()
        
    def deepcopy(self):
        ret = Transition(self.state, self.read_sym, self.n_states, self.output_alph, self.valid_dirs)
        ret.locked, ret.min_w, ret.max_w = self.locked, self.min_w, self.max_w
        
        ret.sym_ind_map, ret.dir_ind_map = self.sym_ind_map, self.dir_ind_map
        
        ret.w_ind = self.w_ind
        ret.total = self.total
        ret.action_totals = self.action_totals
        ret.MLtrans = self.argmax()
        
        return ret
    
    def renormalize(self):
        # prevent stability issues by calling this occasionally
        self.w_ind -= np.log(self.total)
        self.total = 1.
        self.update_action_totals()
        self.MLtrans = self.argmax()
    
    def update_action_totals(self):
        self.action_totals = np.zeros((len(self.output_alph), len(self.valid_dirs)))
        for i in range(len(self.output_alph)):
            for j in range(len(self.valid_dirs)):
                for k in range(self.n_states):
                    self.action_totals[i][j] += np.exp(self.w_ind[k][i][j])
                    
    def prob_of_trans(self, state, action):
        # probability of transitioning to state and outputting action if
        # sampling from the entire distribution
        s_ind = self.sym_ind_map[action.symbol]
        d_ind = self.dir_ind_map[action.direction]
        return np.exp(self.w_ind[state][s_ind][d_ind]) / self.total
                    
    def prob_of_action(self, action):
        # probability of a given action occurring
        s_ind = self.sym_ind_map[action.symbol]
        d_ind = self.dir_ind_map[action.direction]
        return self.action_totals[s_ind][d_ind] / self.total
        
    def probs_all(self):
        # the entire probability distribution
        return np.exp(self.w_ind)/ self.total
                    
    def probs(self, action):
        # probability distribution over states, given an action
        s_ind = self.sym_ind_map[action.symbol]
        d_ind = self.dir_ind_map[action.direction]
        unnormalized = np.exp(self.w_ind[:,s_ind,d_ind])
        return unnormalized / self.action_totals[s_ind][d_ind]
        
    def prob_given_action(self, state, action):
        # probability of transition to a specified state, given an action
        s_ind = self.sym_ind_map[action.symbol]
        d_ind = self.dir_ind_map[action.direction]
        return np.exp(self.get_w(state, action)) / self.action_totals[s_ind][d_ind]
        
    def sample_all(self):
        # sample an action from complete distribution
        p = np.flatten(self.probs_all())
        ind = np.random.choice(self.n_states * len(self.output_alph) * len(self.valid_dirs), p)
        d_ind = ind % len(self.valid_dirs)
        s_ind = (ind / len(self.valid_dirs)) % len(self.output_alph)
        state = ind / (len(self.valid_dirs) * len(self.output_alph))
        return (state, self.output_alph[s_ind], self.valid_dirs[d_ind])
        
    def sample(self, action):
        # sample a state given an action
        p = np.flatten(self.probs(action.symbol, action.direction))
        state = np.random.choice(self.n_states, p)
        return state
        
    def argmax(self):
        # maximum likelihood transition
        best_action = None
        best_state = None
        best_w = float('-inf')
        for i in range(self.n_states):
            for j in range(len(self.output_alph)):
                for k in range(len(self.valid_dirs)):
                    if self.w_ind[i][j][k] > best_w:
                        best_w = self.w_ind[i][j][k]
                        best_state= i
                        best_action = Action(self.output_alph[j], self.valid_dirs[k])
        return best_state, best_action
        
    def get_w(self, state, action):
        s_ind = self.sym_ind_map[action.symbol]
        d_ind = self.dir_ind_map[action.direction]
        return self.w_ind[state][s_ind][d_ind]
    
    def upd_w(self, state, action, val):
        #if self.locked:
        #    return False
        s_ind = self.sym_ind_map[action.symbol]
        d_ind = self.dir_ind_map[action.direction]
        temp = self.w_ind[state][s_ind][d_ind]
        
        if val > temp and self.prob_of_trans(state, action) > 0.999:
            # for numerical stability
            self.locked = True
            return
        if val < temp and self.prob_of_trans(state, action) < 1.e-6:
            # for stability
            return
        
        self.w_ind[state][s_ind][d_ind] = val
        self.action_totals[s_ind][d_ind] += (np.exp(val) - np.exp(temp))
        self.total += (np.exp(val) - np.exp(temp))
        if np.isnan(self.total):
            print self.total()
            sys.exit()
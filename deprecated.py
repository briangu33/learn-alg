# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:00:55 2017

@author: brian
"""

class TransitionWeights():
    # helper class to store log likelihoods of all possible transitions
    
    def __init__(self, n_states, input_alph, output_alph, valid_dirs):
        self.n_states = n_states
        self.input_alphabet = input_alph
        self.output_alphabet = output_alph
        self.valid_dirs = valid_dirs

        # w_transit_to[i][s1][j][s2][d] is the log likelihood that, if PTM
        # begins in state i and reads input symbol s1, then it wlil transition
        # to state s2, output symbol j, and move input tape head in direction d        
        
        self.w_transit_to = []
        for i in range(self.n_states):
            self.w_transit_to.append(dict())
            for s1 in self.input_alphabet:
                self.w_transit_to[i][s1] = []
                for j in range(self.n_states):
                    self.w_transit_to[i][s1].append(dict())
                    for s2 in self.output_alphabet:
                        self.w_transit_to[i][s1][j][s2] = dict()
                        for d in valid_dirs:
                            self.w_transit_to[i][s1][j][s2][d] \
                                = np.random.normal(0, 0.2)
    
    def __call__(self, i, s1, j, s2, d):
        return self.w_transit_to[i][s1][j][s2][d]
        
    def update(self, i, s1, j, s2, d, val):
        self.w_transit_to[i][s1][j][s2][d] = val
                                
    def deep_copy(self):
        copy = TransitionWeights(self.n_states, self.input_alphabet,
                                 self.output_alphabet, self.valid_dirs)
        for i in range(self.n_states):
            for s1 in self.input_alphabet:
                for j in range(self.n_states):
                    for s2 in self.output_alphabet:
                        for d in valid_dirs:
                            copy.w_transit_to[i][s1][j][s2][d] \
                                = self.w_transit_to[i][s1][j][s2][d]
        
        return copy
        
    def p_transit(self, i, s1, j, s2, d):
        # probability that, from state i reading input s1, we transition to 
        # state j, output s2, and move in direction d
        z = 0. # normalization constant
        for next_state in range(self.n_states):
            for symbol in self.output_alphabet:
                for direction in valid_dirs:
                    z += np.exp(self.w_transit_to[i][s1][next_state][symbol][direction])
        return np.exp(self.w_transit_to[i][s1][j][s2][d]) / z
        
    def p_transit_act(self, i, s1, j, act):
        # probability that, from state i reading input s1, given the action
        # (act), we transition to state j
        z = 0. # normalization constant
        for next_state in range(self.n_states):
            z += np.exp(self.w_transit_to[i][s1][next_state][act.symbol][act.direction])
        return np.exp(self.w_transit_to[i][s1][j][act.symbol][act.direction]) / z
        
    def renormalize(self):
        total = 0.
        for i in range(self.n_states):
            for s1 in self.input_alphabet:
                for j in range(self.n_states):
                    for s2 in self.output_alphabet:
                        for d in valid_dirs:
                            total += self.w_transit_to[i][s1][j][s2][d]
        per_weight = total / (self.n_states * len(self.input_alphabet) * self.n_states * len(self.output_alphabet) * len(self.valid_dirs))
        for i in range(self.n_states):
            for s1 in self.input_alphabet:
                for j in range(self.n_states):
                    for s2 in self.output_alphabet:
                        for d in valid_dirs:
                            self.w_transit_to[i][s1][j][s2][d] -= per_weight
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 00:04:24 2017

@author: brian
"""

from action import Action
from action import Direction
from action import Symbol
from ptm import PTMControl
from ptm import constant_learn_rate
from training_gen import input_gen
from training_gen import target_actions_reverse
from training_gen import copy_input_gen
from training_gen import copy_target_actions
from training_gen import walk_gen
from tape import InputTape1D
from tape import InputTape2D

import math
import numpy as np
import sys
import time

n_states = 2
input_alphabet = [Symbol('u'), Symbol('d'), Symbol('l'), Symbol('r'), Symbol('e')]
# valid_dirs = [Direction('left'), Direction('right')]
valid_dirs = [Direction('left'), Direction('right'), Direction('up'), Direction('down')]
ptmctrl = PTMControl(n_states, input_alphabet, valid_dirs, learn_sched = constant_learn_rate)

avg_log_prob_start = 0.
for i in range(20):
    s, acts = walk_gen()
    # acts = target_actions_reverse(s)
    # acts = copy_target_actions(s)
    input_tape = InputTape2D(s)
    prob = ptmctrl.run2D(input_tape, acts)
    # print prob
    avg_log_prob_start += (math.log(prob) / len(acts))
avg_log_prob_start /= 20

print avg_log_prob_start
print "************************"

#learned = []
count = 0
available_states = [0]

for i in range(15000):
    s, acts = walk_gen()
    # acts = target_actions_reverse(s)
    # acts = copy_target_actions(s)
    input_tape = InputTape2D(s)
    prob = ptmctrl.run2D(input_tape, acts, train = True)
    #if not(learn is None):
    #    learned.append(learn)
    #    count += 1
    if i == 0:
        smooth = np.log(prob) / len(acts)
    smooth = 0.95 * smooth + 0.05 * np.log(prob) / len(acts)
    if i % 10 == 0:
        print smooth
    
    if i % 100 == 0:
        for q in range(ptmctrl.n_states):
            for s1 in ptmctrl.input_alphabet:
                j, a = ptmctrl.w[q][s1].argmax()
                s2 = a.symbol
                d = a.direction
                p = ptmctrl.w[q][s1].prob_of_trans(j, a)
                print q, "|", s1, "|", j, "|", s2.rep if not(s2.is_term) else "t", "|", d.name, "|", p

#for learn in learned:
#    print learn.state, learn.read_sym.rep if not(learn.read_sym.is_term) else "t"


if i % 100 == 0:
    for q in range(ptmctrl.n_states):
        for s1 in ptmctrl.input_alphabet:
            j, a = ptmctrl.w[q][s1].argmax()
            s2 = a.symbol
            d = a.direction
            p = ptmctrl.w[q][s1].prob_of_trans(j, a)
            print q, "|", s1, "|", j, "|", s2.rep if not(s2.is_term) else "t", "|", d.name
print "started at:", avg_log_prob_start

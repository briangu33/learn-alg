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

n_states = 5
# input_alphabet = [Symbol('u'), Symbol('d'), Symbol('l'), Symbol('r'), Symbol('e')]
input_alphabet = [Symbol('s'), Symbol('0'), Symbol('1'), Symbol('e')]
valid_dirs = [Direction('left'), Direction('right')]
# valid_dirs = [Direction('left'), Direction('right'), Direction('up'), Direction('down')]
ptmctrl = PTMControl(n_states, input_alphabet, valid_dirs, learn_sched = constant_learn_rate)

symbol0 = Symbol('0')
symbol1 = Symbol('1')
symbolt = Symbol.term_symbol()
symbols = Symbol.start_symbol()
symbole = Symbol.end_symbol()
symbol_ = Symbol.empty_symbol()

dirR = Direction('right')
dirL = Direction('left')
"""
ptmctrl.w[0][symbols].upd_w(0, Action(symbol_, dirR), 0.2)

ptmctrl.w[0][symbol0].upd_w(0, Action(symbol_, dirR), 0.2)

ptmctrl.w[0][symbol1].upd_w(0, Action(symbol_, dirR), 0.)

ptmctrl.w[0][symbole].upd_w(1, Action(symbol_, dirL), 0.2)

ptmctrl.w[1][symbols].upd_w(1, Action(symbolt, dirL), 0.2)

ptmctrl.w[1][symbol0].upd_w(1, Action(symbol0, dirL), 0.2)

ptmctrl.w[1][symbol1].upd_w(1, Action(symbol1, dirL), 0.)

ptmctrl.w[1][symbole].upd_w(1, Action(symbol_, dirL), 0.2)
"""

ptmctrl.w[0][symbols].renormalize()
ptmctrl.w[0][symbol0].renormalize()
ptmctrl.w[0][symbol1].renormalize()
ptmctrl.w[0][symbole].renormalize()
ptmctrl.w[1][symbols].renormalize()
ptmctrl.w[1][symbol0].renormalize()
ptmctrl.w[1][symbol1].renormalize()
ptmctrl.w[1][symbole].renormalize()

avg_log_prob_start = 0.
for i in range(20):
    s = copy_input_gen()
    # s, acts = walk_gen()
    acts = target_actions_reverse(s)
    # acts = copy_target_actions(s)
    input_tape = InputTape1D(s)
    prob = ptmctrl.run(input_tape, acts)
    # print prob
    avg_log_prob_start += (math.log(prob) / len(acts))
avg_log_prob_start /= 20

print avg_log_prob_start
print "************************"

#learned = []
count = 0
available_states = [0]

for i in range(100000):
    s = copy_input_gen()
    # s, acts = walk_gen()
    acts = target_actions_reverse(s)
    # acts = copy_target_actions(s)
    input_tape = InputTape1D(s)
    prob = ptmctrl.run(input_tape, acts, train = True)
    #if not(learn is None):
    #    learned.append(learn)
    #    count += 1
    if i == 0:
        smooth = np.log(prob) / len(acts)
    smooth = 0.99 * smooth + 0.01 * np.log(prob) / len(acts)
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

s = copy_input_gen()
acts = target_actions_reverse(s)
input_tape = InputTape1D(s)
ptmctrl.run(input_tape, acts, verbose=True)

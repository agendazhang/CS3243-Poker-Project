from time import sleep
import pprint
from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.engine.card import Card
from pypokerengine.utils.card_utils import gen_cards, _montecarlo_simulation, estimate_hole_card_win_rate, _pick_unused_card
from pypokerengine.engine.action_checker import ActionChecker
from pypokerengine.engine.player import Player
import inspect
from pypokerengine.engine.poker_constants import PokerConstants
import numpy as np
import copy
import random as rand
import math

#This is the preflop strategy that is referenced from University of Alberta Cepheus Poker Project.
#It is a lookup table that determines the strength of the hand (and subsequently next move that the player
#is going to make) based entirely on the 2 starting cards that the player has and the moves between the
#two players in the preflop stage.
#outer dictionary: key is the moves history, value is the inner dictionary for each moves history
#inner dictionary: key is a tuple of (first_card, second_card, same_suit), value is the probability of the actions

no_action_dict = {('9', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.0008, 'prob_raise': 0.9992}, ('5', '5', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0009, 'prob_raise': 0.9991}, ('T', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '5', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('3', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '5', True): {'prob_fold': 0.0, 'prob_call': 0.0004, 'prob_raise': 0.9996}, ('7', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0001, 'prob_raise': 0.9999}, ('5', '6', False): {'prob_fold': 0.0, 'prob_call': 0.001, 'prob_raise': 0.999}, ('8', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0002, 'prob_raise': 0.9998}, ('2', '8', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('2', 'T', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('9', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.0006, 'prob_raise': 0.9994}, ('8', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0012, 'prob_raise': 0.9988}, ('6', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0002, 'prob_raise': 0.9998}, ('3', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '5', True): {'prob_fold': 0.0, 'prob_call': 0.0027, 'prob_raise': 0.9973}, ('J', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '4', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('2', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0004, 'prob_raise': 0.9996}, ('2', '9', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('T', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.0004, 'prob_raise': 0.9996}, ('4', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0004, 'prob_raise': 0.9996}, ('A', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0051, 'prob_raise': 0.9949}, ('K', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', '4', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.001, 'prob_raise': 0.999}, ('2', '3', True): {'prob_fold': 0.0, 'prob_call': 0.0034, 'prob_raise': 0.9966}, ('4', '6', False): {'prob_fold': 0.0, 'prob_call': 0.0012, 'prob_raise': 0.9988}, ('4', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.0014, 'prob_raise': 0.9986}, ('Q', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('T', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0004, 'prob_raise': 0.9996}, ('5', '9', True): {'prob_fold': 0.0, 'prob_call': 0.0004, 'prob_raise': 0.9996}, ('7', '9', False): {'prob_fold': 0.0, 'prob_call': 0.0005, 'prob_raise': 0.9995}, ('9', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0004, 'prob_raise': 0.9996}, ('5', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0003, 'prob_raise': 0.9997}, ('8', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.0008, 'prob_raise': 0.9992}, ('2', '4', True): {'prob_fold': 0.0, 'prob_call': 0.0021, 'prob_raise': 0.9979}, ('Q', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '9', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('6', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0007, 'prob_raise': 0.9993}, ('6', '8', False): {'prob_fold': 0.0, 'prob_call': 0.001, 'prob_raise': 0.999}, ('4', '8', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('Q', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('T', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0008, 'prob_raise': 0.9992}, ('7', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.0008, 'prob_raise': 0.9992}, ('3', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0001, 'prob_raise': 0.9999}, ('4', '7', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0005, 'prob_raise': 0.9995}, ('6', '7', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0005, 'prob_raise': 0.9995}, ('2', 'J', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('5', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.0013, 'prob_raise': 0.9987}, ('7', '8', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0012, 'prob_raise': 0.9988}, ('4', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', '8', True): {'prob_fold': 0.0, 'prob_call': 0.0005, 'prob_raise': 0.9995}, ('6', '9', False): {'prob_fold': 0.0, 'prob_call': 0.001, 'prob_raise': 0.999}, ('3', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0014, 'prob_raise': 0.9986}, ('3', '8', True): {'prob_fold': 0.0, 'prob_call': 0.0023, 'prob_raise': 0.9977}, ('9', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0002, 'prob_raise': 0.9998}, ('2', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', '9', True): {'prob_fold': 0.0, 'prob_call': 0.0012, 'prob_raise': 0.9988}, ('6', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0006, 'prob_raise': 0.9994}, ('J', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('9', '9', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0006, 'prob_raise': 0.9994}, ('2', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0009, 'prob_raise': 0.9991}, ('6', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0005, 'prob_raise': 0.9995}, ('2', '6', True): {'prob_fold': 0.0, 'prob_call': 0.0048, 'prob_raise': 0.9952}, ('7', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0007, 'prob_raise': 0.9993}, ('3', '7', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('J', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0007, 'prob_raise': 0.9993}, ('7', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0003, 'prob_raise': 0.9997}, ('J', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0005, 'prob_raise': 0.9995}, ('4', '5', True): {'prob_fold': 0.0, 'prob_call': 0.0006, 'prob_raise': 0.9994}, ('6', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('9', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0003, 'prob_raise': 0.9997}, ('8', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', '7', True): {'prob_fold': 0.0, 'prob_call': 0.0004, 'prob_raise': 0.9996}, ('5', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0001, 'prob_raise': 0.9999}, ('8', '9', False): {'prob_fold': 0.0, 'prob_call': 0.0007, 'prob_raise': 0.9993}, ('9', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('K', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0002, 'prob_raise': 0.9998}, ('3', '6', True): {'prob_fold': 0.0, 'prob_call': 0.0021, 'prob_raise': 0.9979}, ('2', '7', True): {'prob_fold': 0.0, 'prob_call': 0.0042, 'prob_raise': 0.9958}, ('6', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.0012, 'prob_raise': 0.9988}, ('7', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0005, 'prob_raise': 0.9995}, ('2', '8', True): {'prob_fold': 0.0, 'prob_call': 0.0033, 'prob_raise': 0.9967}, ('3', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0017, 'prob_raise': 0.9983}, ('2', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0035, 'prob_raise': 0.9965}, ('3', '5', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('3', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', '6', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0004, 'prob_raise': 0.9996}, ('3', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.0013, 'prob_raise': 0.9987}, ('9', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0003, 'prob_raise': 0.9997}, ('8', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0003, 'prob_raise': 0.9997}, ('2', '2', False): {'prob_fold': 0.0, 'prob_call': 0.0053, 'prob_raise': 0.9947}, ('2', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0006, 'prob_raise': 0.9994}, ('T', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '4', True): {'prob_fold': 0.0, 'prob_call': 0.0015, 'prob_raise': 0.9985}, ('2', '9', True): {'prob_fold': 0.0, 'prob_call': 0.0037, 'prob_raise': 0.9963}, ('T', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', 'T', False): {'prob_fold': 0.2516, 'prob_call': 0.0019, 'prob_raise': 0.7465}, ('J', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0003, 'prob_raise': 0.9997}, ('2', '3', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('4', '6', True): {'prob_fold': 0.0, 'prob_call': 0.0006, 'prob_raise': 0.9994}, ('5', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0008, 'prob_raise': 0.9992}, ('4', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0003, 'prob_raise': 0.9997}, ('3', '3', False): {'prob_fold': 0.0, 'prob_call': 0.0012, 'prob_raise': 0.9988}, ('6', '6', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', '9', True): {'prob_fold': 0.0, 'prob_call': 0.0003, 'prob_raise': 0.9997}, ('9', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0003, 'prob_raise': 0.9997}, ('Q', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0008, 'prob_raise': 0.9992}, ('T', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '9', True): {'prob_fold': 0.0, 'prob_call': 0.0022, 'prob_raise': 0.9978}, ('5', '9', False): {'prob_fold': 0.0, 'prob_call': 0.0017, 'prob_raise': 0.9983}, ('6', '8', True): {'prob_fold': 0.0, 'prob_call': 0.0003, 'prob_raise': 0.9997}, ('4', '8', True): {'prob_fold': 0.0, 'prob_call': 0.0005, 'prob_raise': 0.9995}, ('2', '4', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('T', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', '7', False): {'prob_fold': 0.9448, 'prob_call': 0.0003, 'prob_raise': 0.0549}, ('4', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0007, 'prob_raise': 0.9993}, ('6', '7', False): {'prob_fold': 0.0, 'prob_call': 0.0006, 'prob_raise': 0.9994}, ('Q', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0004, 'prob_raise': 0.9996}, ('2', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0023, 'prob_raise': 0.9977}, ('7', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0002, 'prob_raise': 0.9998}, ('7', '8', False): {'prob_fold': 0.0, 'prob_call': 0.0007, 'prob_raise': 0.9993}, ('5', '8', False): {'prob_fold': 0.0, 'prob_call': 0.0015, 'prob_raise': 0.9985}, ('8', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0002, 'prob_raise': 0.9998}, ('3', 'T', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('5', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0009, 'prob_raise': 0.9991}, ('3', '8', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('5', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.002, 'prob_raise': 0.998}, ('9', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.0007, 'prob_raise': 0.9993}, ('2', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0016, 'prob_raise': 0.9984}, ('4', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', '9', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('6', '9', True): {'prob_fold': 0.0, 'prob_call': 0.0005, 'prob_raise': 0.9995}, ('J', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('K', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0002, 'prob_raise': 0.9998}, ('6', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.0012, 'prob_raise': 0.9988}, ('7', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0001, 'prob_raise': 0.9999}, ('3', '7', True): {'prob_fold': 0.0, 'prob_call': 0.0004, 'prob_raise': 0.9996}, ('2', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0009, 'prob_raise': 0.9991}, ('8', '8', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0001, 'prob_raise': 0.9999}, ('2', '6', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('5', '7', False): {'prob_fold': 0.0, 'prob_call': 0.0007, 'prob_raise': 0.9993}, ('4', '5', False): {'prob_fold': 0.0, 'prob_call': 0.0009, 'prob_raise': 0.9991}, ('9', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0002, 'prob_raise': 0.9998}, ('J', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('T', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0004, 'prob_raise': 0.9996}, ('5', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.0008, 'prob_raise': 0.9992}, ('5', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '6', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('7', '7', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '7', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('6', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0006, 'prob_raise': 0.9994}, ('8', '9', True): {'prob_fold': 0.0, 'prob_call': 0.0001, 'prob_raise': 0.9999}}

raise_dict = {('9', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.6175, 'prob_raise': 0.3825}, ('7', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.8197, 'prob_raise': 0.1803}, ('5', '5', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '5', False): {'prob_fold': 0.8626, 'prob_call': 0.1374, 'prob_raise': 0.0}, ('3', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.7708, 'prob_raise': 0.2292}, ('3', '5', True): {'prob_fold': 0.0, 'prob_call': 0.3208, 'prob_raise': 0.6792}, ('7', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', '6', False): {'prob_fold': 0.0, 'prob_call': 0.903, 'prob_raise': 0.097}, ('8', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '8', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('2', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.7268, 'prob_raise': 0.2732}, ('8', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.4248, 'prob_raise': 0.5752}, ('6', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0035, 'prob_raise': 0.9965}, ('3', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.7624, 'prob_raise': 0.2376}, ('2', '5', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '4', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.9981, 'prob_raise': 0.0019}, ('2', '9', False): {'prob_fold': 0.3121, 'prob_call': 0.6879, 'prob_raise': 0.0}, ('T', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.6334, 'prob_raise': 0.3666}, ('4', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.5909, 'prob_raise': 0.4091}, ('A', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('K', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', '4', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.3876, 'prob_raise': 0.6124}, ('4', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.4057, 'prob_raise': 0.5943}, ('2', '3', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('T', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.5049, 'prob_raise': 0.4951}, ('5', '9', True): {'prob_fold': 0.0, 'prob_call': 0.383, 'prob_raise': 0.617}, ('7', '9', False): {'prob_fold': 0.0, 'prob_call': 0.8616, 'prob_raise': 0.1384}, ('9', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.2246, 'prob_raise': 0.7754}, ('5', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.8542, 'prob_raise': 0.1458}, ('8', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.7565, 'prob_raise': 0.2435}, ('2', '4', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.7733, 'prob_raise': 0.2267}, ('6', '8', False): {'prob_fold': 0.0, 'prob_call': 0.9826, 'prob_raise': 0.0174}, ('4', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('T', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.8474, 'prob_raise': 0.1526}, ('3', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '7', True): {'prob_fold': 0.0, 'prob_call': 0.579, 'prob_raise': 0.421}, ('4', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.3941, 'prob_raise': 0.6059}, ('6', '7', True): {'prob_fold': 0.0, 'prob_call': 0.0094, 'prob_raise': 0.9906}, ('8', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.8469, 'prob_raise': 0.1531}, ('2', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.9689, 'prob_raise': 0.0311}, ('7', '8', True): {'prob_fold': 0.0, 'prob_call': 0.0624, 'prob_raise': 0.9376}, ('4', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.9868, 'prob_raise': 0.0132}, ('4', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.9915, 'prob_raise': 0.0085}, ('5', '8', True): {'prob_fold': 0.0, 'prob_call': 0.3692, 'prob_raise': 0.6308}, ('6', '9', False): {'prob_fold': 0.0, 'prob_call': 0.9698, 'prob_raise': 0.0302}, ('3', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.8442, 'prob_raise': 0.1558}, ('3', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0337, 'prob_raise': 0.9663}, ('2', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.9943, 'prob_raise': 0.0057}, ('4', '9', True): {'prob_fold': 0.0, 'prob_call': 0.9846, 'prob_raise': 0.0154}, ('6', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.2544, 'prob_raise': 0.7456}, ('J', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0119, 'prob_raise': 0.9881}, ('9', '9', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.5759, 'prob_raise': 0.4241}, ('2', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.7975, 'prob_raise': 0.2025}, ('2', '6', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.8661, 'prob_raise': 0.1339}, ('3', '7', False): {'prob_fold': 0.7351, 'prob_call': 0.2649, 'prob_raise': 0.0}, ('J', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.7589, 'prob_raise': 0.2411}, ('J', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.3849, 'prob_raise': 0.6151}, ('4', '5', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('9', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.7669, 'prob_raise': 0.2331}, ('8', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.2076, 'prob_raise': 0.7924}, ('5', '7', True): {'prob_fold': 0.0, 'prob_call': 0.1298, 'prob_raise': 0.8702}, ('5', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.7282, 'prob_raise': 0.2718}, ('8', '9', False): {'prob_fold': 0.0, 'prob_call': 0.6748, 'prob_raise': 0.3252}, ('9', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('K', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.8414, 'prob_raise': 0.1586}, ('3', '6', True): {'prob_fold': 0.0, 'prob_call': 0.7882, 'prob_raise': 0.2118}, ('2', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.9812, 'prob_raise': 0.0188}, ('7', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.3946, 'prob_raise': 0.6054}, ('8', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.1517, 'prob_raise': 0.8483}, ('8', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.8159, 'prob_raise': 0.1841}, ('2', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.9917, 'prob_raise': 0.0083}, ('3', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.6242, 'prob_raise': 0.3758}, ('5', '6', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.7575, 'prob_raise': 0.2425}, ('3', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0387, 'prob_raise': 0.9613}, ('8', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '2', False): {'prob_fold': 0.0, 'prob_call': 0.5435, 'prob_raise': 0.4565}, ('2', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '4', True): {'prob_fold': 0.0, 'prob_call': 0.3214, 'prob_raise': 0.6786}, ('2', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.5786, 'prob_raise': 0.4214}, ('2', '3', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('4', '6', True): {'prob_fold': 0.0, 'prob_call': 0.1358, 'prob_raise': 0.8642}, ('5', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.8472, 'prob_raise': 0.1528}, ('4', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.47, 'prob_raise': 0.53}, ('3', '3', False): {'prob_fold': 0.0, 'prob_call': 0.1657, 'prob_raise': 0.8343}, ('6', '6', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', '9', True): {'prob_fold': 0.0, 'prob_call': 0.1663, 'prob_raise': 0.8337}, ('9', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.1522, 'prob_raise': 0.8478}, ('Q', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('T', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '9', True): {'prob_fold': 0.0, 'prob_call': 0.9695, 'prob_raise': 0.0305}, ('5', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '8', True): {'prob_fold': 0.0, 'prob_call': 0.1129, 'prob_raise': 0.8871}, ('4', '8', True): {'prob_fold': 0.0, 'prob_call': 0.864, 'prob_raise': 0.136}, ('2', '4', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('T', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0745, 'prob_raise': 0.9255}, ('6', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.3552, 'prob_raise': 0.6448}, ('4', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.9517, 'prob_raise': 0.0483}, ('6', '7', False): {'prob_fold': 0.0, 'prob_call': 0.8305, 'prob_raise': 0.1695}, ('Q', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.2358, 'prob_raise': 0.7642}, ('2', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.9774, 'prob_raise': 0.0226}, ('7', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.2172, 'prob_raise': 0.7828}, ('7', '8', False): {'prob_fold': 0.0, 'prob_call': 0.8014, 'prob_raise': 0.1986}, ('5', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0709, 'prob_raise': 0.9291}, ('3', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.352, 'prob_raise': 0.648}, ('3', '8', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('5', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.6075, 'prob_raise': 0.3925}, ('2', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '9', True): {'prob_fold': 0.0, 'prob_call': 0.1815, 'prob_raise': 0.8185}, ('J', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('K', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0005, 'prob_raise': 0.9995}, ('6', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.9793, 'prob_raise': 0.0207}, ('7', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.3648, 'prob_raise': 0.6352}, ('3', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', '8', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '6', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('5', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '5', False): {'prob_fold': 0.0, 'prob_call': 0.9602, 'prob_raise': 0.0398}, ('9', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('J', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('T', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.6871, 'prob_raise': 0.3129}, ('5', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.776, 'prob_raise': 0.224}, ('5', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.7276, 'prob_raise': 0.2724}, ('3', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '7', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '7', False): {'prob_fold': 1.0, 'prob_call': 0.0, 'prob_raise': 0.0}, ('6', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.5754, 'prob_raise': 0.4246}, ('8', '9', True): {'prob_fold': 0.0, 'prob_call': 0.0774, 'prob_raise': 0.9226}}

raise_raise_dict = {('9', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.9999, 'prob_raise': 0.0001}, ('5', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.9998, 'prob_raise': 0.0002}, ('3', '5', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.9995, 'prob_raise': 0.0005}, ('5', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '5', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '4', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.9997, 'prob_raise': 0.0003}, ('A', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('K', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '4', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.9999, 'prob_raise': 0.0001}, ('2', '3', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.9998, 'prob_raise': 0.0002}, ('T', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.9998, 'prob_raise': 0.0002}, ('8', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '4', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.9998, 'prob_raise': 0.0002}, ('6', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.9999, 'prob_raise': 0.0001}, ('7', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.9997, 'prob_raise': 0.0003}, ('2', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '8', True): {'prob_fold': 0.0, 'prob_call': 0.9996, 'prob_raise': 0.0004}, ('4', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.9999, 'prob_raise': 0.0001}, ('5', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.9994, 'prob_raise': 0.0006}, ('2', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.9999, 'prob_raise': 0.0001}, ('2', '6', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.9998, 'prob_raise': 0.0002}, ('J', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '5', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.9998, 'prob_raise': 0.0002}, ('8', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.9998, 'prob_raise': 0.0002}, ('8', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('K', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.9998, 'prob_raise': 0.0002}, ('3', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '6', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.9995, 'prob_raise': 0.0005}, ('8', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.9998, 'prob_raise': 0.0002}, ('5', '6', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.9997, 'prob_raise': 0.0003}, ('2', '2', False): {'prob_fold': 0.0, 'prob_call': 0.9994, 'prob_raise': 0.0006}, ('2', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.9998, 'prob_raise': 0.0002}, ('3', '4', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.9995, 'prob_raise': 0.0005}, ('4', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.9998, 'prob_raise': 0.0002}, ('2', '3', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '6', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.9999, 'prob_raise': 0.0001}, ('4', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.9998, 'prob_raise': 0.0002}, ('3', '3', False): {'prob_fold': 0.0, 'prob_call': 0.9997, 'prob_raise': 0.0003}, ('6', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '9', True): {'prob_fold': 0.0, 'prob_call': 0.9997, 'prob_raise': 0.0003}, ('9', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.9999, 'prob_raise': 0.0001}, ('3', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '4', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.9999, 'prob_raise': 0.0001}, ('6', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.9999, 'prob_raise': 0.0001}, ('7', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.9998, 'prob_raise': 0.0002}, ('2', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.9999, 'prob_raise': 0.0001}, ('4', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '9', True): {'prob_fold': 0.0, 'prob_call': 0.9998, 'prob_raise': 0.0002}, ('J', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.9998, 'prob_raise': 0.0002}, ('K', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.9999, 'prob_raise': 0.0001}, ('2', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.9998, 'prob_raise': 0.0002}, ('5', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '7', False): {'prob_fold': 0.0, 'prob_call': 0.9999, 'prob_raise': 0.0001}, ('2', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', '9', True): {'prob_fold': 0.0, 'prob_call': 0.9999, 'prob_raise': 0.0001}}

raise_raise_raise_dict = {('9', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '5', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '5', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '4', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('A', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('K', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '4', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '3', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '4', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '6', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '5', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('K', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '6', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '6', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '2', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '4', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '3', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '6', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '3', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '4', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('K', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}}

call_dict = {('9', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0141, 'prob_raise': 0.9859}, ('7', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.5786, 'prob_raise': 0.4214}, ('5', '5', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.7516, 'prob_raise': 0.2484}, ('T', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.1672, 'prob_raise': 0.8328}, ('3', '5', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', '6', False): {'prob_fold': 0.0, 'prob_call': 0.3887, 'prob_raise': 0.6113}, ('8', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.1446, 'prob_raise': 0.8554}, ('8', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.3053, 'prob_raise': 0.6947}, ('2', '5', True): {'prob_fold': 0.0, 'prob_call': 0.5105, 'prob_raise': 0.4895}, ('J', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '4', False): {'prob_fold': 0.0, 'prob_call': 0.7976, 'prob_raise': 0.2024}, ('2', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.0052, 'prob_raise': 0.9948}, ('4', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.2901, 'prob_raise': 0.7099}, ('A', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('K', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', '4', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '3', True): {'prob_fold': 0.0, 'prob_call': 0.7207, 'prob_raise': 0.2793}, ('4', '6', False): {'prob_fold': 0.0, 'prob_call': 0.5954, 'prob_raise': 0.4046}, ('4', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.7252, 'prob_raise': 0.2748}, ('Q', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('T', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', '9', True): {'prob_fold': 0.0, 'prob_call': 0.1374, 'prob_raise': 0.8626}, ('7', '9', False): {'prob_fold': 0.0, 'prob_call': 0.4103, 'prob_raise': 0.5897}, ('9', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.3934, 'prob_raise': 0.6066}, ('8', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.2791, 'prob_raise': 0.7209}, ('2', '4', True): {'prob_fold': 0.0, 'prob_call': 0.5783, 'prob_raise': 0.4217}, ('Q', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.4943, 'prob_raise': 0.5057}, ('6', '8', False): {'prob_fold': 0.0, 'prob_call': 0.49, 'prob_raise': 0.51}, ('4', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('T', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.4815, 'prob_raise': 0.5185}, ('3', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.4387, 'prob_raise': 0.5613}, ('4', '7', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', '7', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.355, 'prob_raise': 0.645}, ('2', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.6338, 'prob_raise': 0.3662}, ('7', '8', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.624, 'prob_raise': 0.376}, ('4', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.2317, 'prob_raise': 0.7683}, ('5', '8', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', '9', False): {'prob_fold': 0.0, 'prob_call': 0.6164, 'prob_raise': 0.3836}, ('3', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.45, 'prob_raise': 0.55}, ('3', '8', True): {'prob_fold': 0.0, 'prob_call': 0.7929, 'prob_raise': 0.2071}, ('9', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.4247, 'prob_raise': 0.5753}, ('4', '9', True): {'prob_fold': 0.0, 'prob_call': 0.5193, 'prob_raise': 0.4807}, ('6', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('J', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('9', '9', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0001, 'prob_raise': 0.9999}, ('2', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.957, 'prob_raise': 0.043}, ('6', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.2185, 'prob_raise': 0.7815}, ('2', '6', True): {'prob_fold': 0.0, 'prob_call': 0.7392, 'prob_raise': 0.2608}, ('7', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.5555, 'prob_raise': 0.4445}, ('3', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.2273, 'prob_raise': 0.7727}, ('J', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', '5', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('9', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.1509, 'prob_raise': 0.8491}, ('8', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', '7', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', '9', False): {'prob_fold': 0.0, 'prob_call': 0.1758, 'prob_raise': 0.8242}, ('9', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('K', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.1228, 'prob_raise': 0.8772}, ('3', '6', True): {'prob_fold': 0.0, 'prob_call': 0.2265, 'prob_raise': 0.7735}, ('2', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.7163, 'prob_raise': 0.2837}, ('7', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.2241, 'prob_raise': 0.7759}, ('2', '8', True): {'prob_fold': 0.0, 'prob_call': 0.9609, 'prob_raise': 0.0391}, ('3', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.7746, 'prob_raise': 0.2254}, ('2', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.6192, 'prob_raise': 0.3808}, ('3', '5', False): {'prob_fold': 0.0, 'prob_call': 0.7137, 'prob_raise': 0.2863}, ('3', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', '6', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.2876, 'prob_raise': 0.7124}, ('3', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.9025, 'prob_raise': 0.0975}, ('9', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '2', False): {'prob_fold': 0.0, 'prob_call': 0.8421, 'prob_raise': 0.1579}, ('2', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.6575, 'prob_raise': 0.3425}, ('T', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '4', True): {'prob_fold': 0.0, 'prob_call': 0.0437, 'prob_raise': 0.9563}, ('2', '9', True): {'prob_fold': 0.0, 'prob_call': 0.7972, 'prob_raise': 0.2028}, ('T', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.9009, 'prob_raise': 0.0991}, ('J', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '3', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '6', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.5269, 'prob_raise': 0.4731}, ('4', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0765, 'prob_raise': 0.9235}, ('3', '3', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', '6', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', '9', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('9', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('Q', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('T', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '9', True): {'prob_fold': 0.0, 'prob_call': 0.5979, 'prob_raise': 0.4021}, ('5', '9', False): {'prob_fold': 0.0, 'prob_call': 0.8337, 'prob_raise': 0.1663}, ('6', '8', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', '8', True): {'prob_fold': 0.0, 'prob_call': 0.3067, 'prob_raise': 0.6933}, ('2', '4', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', '7', False): {'prob_fold': 0.0, 'prob_call': 0.7888, 'prob_raise': 0.2112}, ('4', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.5836, 'prob_raise': 0.4164}, ('6', '7', False): {'prob_fold': 0.0, 'prob_call': 0.2884, 'prob_raise': 0.7116}, ('Q', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.4833, 'prob_raise': 0.5167}, ('7', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', '8', False): {'prob_fold': 0.0, 'prob_call': 0.2395, 'prob_raise': 0.7605}, ('5', '8', False): {'prob_fold': 0.0, 'prob_call': 0.7022, 'prob_raise': 0.2978}, ('8', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.8457, 'prob_raise': 0.1543}, ('9', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.0227, 'prob_raise': 0.9773}, ('2', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.9783, 'prob_raise': 0.0217}, ('4', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '9', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('J', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('K', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.6816, 'prob_raise': 0.3184}, ('7', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '7', True): {'prob_fold': 0.0, 'prob_call': 0.5063, 'prob_raise': 0.4937}, ('2', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.3952, 'prob_raise': 0.6048}, ('8', '8', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '7', False): {'prob_fold': 0.0, 'prob_call': 0.5279, 'prob_raise': 0.4721}, ('4', '5', False): {'prob_fold': 0.0, 'prob_call': 0.4383, 'prob_raise': 0.5617}, ('9', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('J', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('T', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0028, 'prob_raise': 0.9972}, ('5', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.3673, 'prob_raise': 0.6327}, ('5', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.2472, 'prob_raise': 0.7528}, ('3', '6', False): {'prob_fold': 0.0, 'prob_call': 0.9806, 'prob_raise': 0.0194}, ('7', '7', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', '9', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}}

call_raise_dict = {('9', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0141, 'prob_raise': 0.9859}, ('7', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.5786, 'prob_raise': 0.4214}, ('5', '5', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.7516, 'prob_raise': 0.2484}, ('T', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.1672, 'prob_raise': 0.8328}, ('3', '5', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', '6', False): {'prob_fold': 0.0, 'prob_call': 0.3887, 'prob_raise': 0.6113}, ('8', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.1446, 'prob_raise': 0.8554}, ('8', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.3053, 'prob_raise': 0.6947}, ('2', '5', True): {'prob_fold': 0.0, 'prob_call': 0.5105, 'prob_raise': 0.4895}, ('J', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '4', False): {'prob_fold': 0.0, 'prob_call': 0.7976, 'prob_raise': 0.2024}, ('2', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.0052, 'prob_raise': 0.9948}, ('4', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.2901, 'prob_raise': 0.7099}, ('A', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('K', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', '4', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '3', True): {'prob_fold': 0.0, 'prob_call': 0.7207, 'prob_raise': 0.2793}, ('4', '6', False): {'prob_fold': 0.0, 'prob_call': 0.5954, 'prob_raise': 0.4046}, ('4', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.7252, 'prob_raise': 0.2748}, ('Q', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('T', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', '9', True): {'prob_fold': 0.0, 'prob_call': 0.1374, 'prob_raise': 0.8626}, ('7', '9', False): {'prob_fold': 0.0, 'prob_call': 0.4103, 'prob_raise': 0.5897}, ('9', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.3934, 'prob_raise': 0.6066}, ('8', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.2791, 'prob_raise': 0.7209}, ('2', '4', True): {'prob_fold': 0.0, 'prob_call': 0.5783, 'prob_raise': 0.4217}, ('Q', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.4943, 'prob_raise': 0.5057}, ('6', '8', False): {'prob_fold': 0.0, 'prob_call': 0.49, 'prob_raise': 0.51}, ('4', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('T', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.4815, 'prob_raise': 0.5185}, ('3', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.4387, 'prob_raise': 0.5613}, ('4', '7', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', '7', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.355, 'prob_raise': 0.645}, ('2', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.6338, 'prob_raise': 0.3662}, ('7', '8', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.624, 'prob_raise': 0.376}, ('4', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.2317, 'prob_raise': 0.7683}, ('5', '8', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', '9', False): {'prob_fold': 0.0, 'prob_call': 0.6164, 'prob_raise': 0.3836}, ('3', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.45, 'prob_raise': 0.55}, ('3', '8', True): {'prob_fold': 0.0, 'prob_call': 0.7929, 'prob_raise': 0.2071}, ('9', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.4247, 'prob_raise': 0.5753}, ('4', '9', True): {'prob_fold': 0.0, 'prob_call': 0.5193, 'prob_raise': 0.4807}, ('6', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('J', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('9', '9', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0001, 'prob_raise': 0.9999}, ('2', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.957, 'prob_raise': 0.043}, ('6', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.2185, 'prob_raise': 0.7815}, ('2', '6', True): {'prob_fold': 0.0, 'prob_call': 0.7392, 'prob_raise': 0.2608}, ('7', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.5555, 'prob_raise': 0.4445}, ('3', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.2273, 'prob_raise': 0.7727}, ('J', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', '5', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('9', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.1509, 'prob_raise': 0.8491}, ('8', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', '7', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', '9', False): {'prob_fold': 0.0, 'prob_call': 0.1758, 'prob_raise': 0.8242}, ('9', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('K', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.1228, 'prob_raise': 0.8772}, ('3', '6', True): {'prob_fold': 0.0, 'prob_call': 0.2265, 'prob_raise': 0.7735}, ('2', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.7163, 'prob_raise': 0.2837}, ('7', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.2241, 'prob_raise': 0.7759}, ('2', '8', True): {'prob_fold': 0.0, 'prob_call': 0.9609, 'prob_raise': 0.0391}, ('3', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.7746, 'prob_raise': 0.2254}, ('2', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.6192, 'prob_raise': 0.3808}, ('3', '5', False): {'prob_fold': 0.0, 'prob_call': 0.7137, 'prob_raise': 0.2863}, ('3', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', '6', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.2876, 'prob_raise': 0.7124}, ('3', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.9025, 'prob_raise': 0.0975}, ('9', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '2', False): {'prob_fold': 0.0, 'prob_call': 0.8421, 'prob_raise': 0.1579}, ('2', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.6575, 'prob_raise': 0.3425}, ('T', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '4', True): {'prob_fold': 0.0, 'prob_call': 0.0437, 'prob_raise': 0.9563}, ('2', '9', True): {'prob_fold': 0.0, 'prob_call': 0.7972, 'prob_raise': 0.2028}, ('T', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.9009, 'prob_raise': 0.0991}, ('J', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '3', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '6', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.5269, 'prob_raise': 0.4731}, ('4', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0765, 'prob_raise': 0.9235}, ('3', '3', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', '6', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', '9', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('9', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('5', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('Q', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('T', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '9', True): {'prob_fold': 0.0, 'prob_call': 0.5979, 'prob_raise': 0.4021}, ('5', '9', False): {'prob_fold': 0.0, 'prob_call': 0.8337, 'prob_raise': 0.1663}, ('6', '8', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', '8', True): {'prob_fold': 0.0, 'prob_call': 0.3067, 'prob_raise': 0.6933}, ('2', '4', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', '7', False): {'prob_fold': 0.0, 'prob_call': 0.7888, 'prob_raise': 0.2112}, ('4', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.5836, 'prob_raise': 0.4164}, ('6', '7', False): {'prob_fold': 0.0, 'prob_call': 0.2884, 'prob_raise': 0.7116}, ('Q', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.4833, 'prob_raise': 0.5167}, ('7', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', '8', False): {'prob_fold': 0.0, 'prob_call': 0.2395, 'prob_raise': 0.7605}, ('5', '8', False): {'prob_fold': 0.0, 'prob_call': 0.7022, 'prob_raise': 0.2978}, ('8', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.8457, 'prob_raise': 0.1543}, ('9', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.0227, 'prob_raise': 0.9773}, ('2', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.9783, 'prob_raise': 0.0217}, ('4', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('4', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '9', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('J', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('K', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('6', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.6816, 'prob_raise': 0.3184}, ('7', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('3', '7', True): {'prob_fold': 0.0, 'prob_call': 0.5063, 'prob_raise': 0.4937}, ('2', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.3952, 'prob_raise': 0.6048}, ('8', '8', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('7', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '7', False): {'prob_fold': 0.0, 'prob_call': 0.5279, 'prob_raise': 0.4721}, ('4', '5', False): {'prob_fold': 0.0, 'prob_call': 0.4383, 'prob_raise': 0.5617}, ('9', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('J', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('T', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.0028, 'prob_raise': 0.9972}, ('5', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.3673, 'prob_raise': 0.6327}, ('5', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.2472, 'prob_raise': 0.7528}, ('3', '6', False): {'prob_fold': 0.0, 'prob_call': 0.9806, 'prob_raise': 0.0194}, ('7', '7', False): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('2', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}, ('8', '9', True): {'prob_fold': 0.0, 'prob_call': 0.0, 'prob_raise': 1.0}}

call_raise_raise_dict = {('9', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.6683, 'prob_raise': 0.3317}, ('7', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.9715, 'prob_raise': 0.0285}, ('5', '5', False): {'prob_fold': 0.0, 'prob_call': 0.4169, 'prob_raise': 0.5831}, ('3', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.5289, 'prob_raise': 0.4711}, ('2', '5', False): {'prob_fold': 0.0059, 'prob_call': 0.9941, 'prob_raise': 0.0}, ('3', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '5', True): {'prob_fold': 0.0, 'prob_call': 0.8508, 'prob_raise': 0.1492}, ('7', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.5296, 'prob_raise': 0.4704}, ('5', '6', False): {'prob_fold': 0.0, 'prob_call': 0.9985, 'prob_raise': 0.0015}, ('8', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.617, 'prob_raise': 0.383}, ('2', '8', False): {'prob_fold': 0.1692, 'prob_call': 0.8308, 'prob_raise': 0.0}, ('2', 'T', False): {'prob_fold': 0.0541, 'prob_call': 0.9459, 'prob_raise': 0.0}, ('9', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.9616, 'prob_raise': 0.0384}, ('8', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.6212, 'prob_raise': 0.3788}, ('6', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.6746, 'prob_raise': 0.3254}, ('3', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '5', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.5076, 'prob_raise': 0.4924}, ('3', '4', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.6672, 'prob_raise': 0.3328}, ('2', '9', False): {'prob_fold': 0.0728, 'prob_call': 0.9272, 'prob_raise': 0.0}, ('T', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.7203, 'prob_raise': 0.2797}, ('4', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('A', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.1806, 'prob_raise': 0.8194}, ('K', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.4342, 'prob_raise': 0.5658}, ('4', '4', False): {'prob_fold': 0.0, 'prob_call': 0.524, 'prob_raise': 0.476}, ('5', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.8972, 'prob_raise': 0.1028}, ('4', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '3', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.4174, 'prob_raise': 0.5826}, ('T', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.5906, 'prob_raise': 0.4094}, ('5', '9', True): {'prob_fold': 0.0, 'prob_call': 0.8947, 'prob_raise': 0.1053}, ('7', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.5722, 'prob_raise': 0.4278}, ('5', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.8925, 'prob_raise': 0.1075}, ('2', '4', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.4784, 'prob_raise': 0.5216}, ('3', '9', False): {'prob_fold': 0.0033, 'prob_call': 0.9967, 'prob_raise': 0.0}, ('6', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.4853, 'prob_raise': 0.5147}, ('T', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.4422, 'prob_raise': 0.5578}, ('7', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.8279, 'prob_raise': 0.1721}, ('4', '7', True): {'prob_fold': 0.0, 'prob_call': 0.8512, 'prob_raise': 0.1488}, ('4', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.7991, 'prob_raise': 0.2009}, ('6', '7', True): {'prob_fold': 0.0, 'prob_call': 0.5917, 'prob_raise': 0.4083}, ('8', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.9901, 'prob_raise': 0.0099}, ('2', 'J', False): {'prob_fold': 0.0298, 'prob_call': 0.9702, 'prob_raise': 0.0}, ('5', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '8', True): {'prob_fold': 0.0, 'prob_call': 0.58, 'prob_raise': 0.42}, ('4', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.7699, 'prob_raise': 0.2301}, ('5', '8', True): {'prob_fold': 0.0, 'prob_call': 0.754, 'prob_raise': 0.246}, ('6', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.756, 'prob_raise': 0.244}, ('2', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.7703, 'prob_raise': 0.2297}, ('J', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.4979, 'prob_raise': 0.5021}, ('9', '9', False): {'prob_fold': 0.0, 'prob_call': 0.1734, 'prob_raise': 0.8266}, ('7', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.6689, 'prob_raise': 0.3311}, ('2', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.7228, 'prob_raise': 0.2772}, ('2', '6', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.4388, 'prob_raise': 0.5612}, ('7', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.8483, 'prob_raise': 0.1517}, ('J', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.5558, 'prob_raise': 0.4442}, ('4', '5', True): {'prob_fold': 0.0, 'prob_call': 0.656, 'prob_raise': 0.344}, ('6', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.5649, 'prob_raise': 0.4351}, ('9', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.7585, 'prob_raise': 0.2415}, ('8', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.7826, 'prob_raise': 0.2174}, ('5', '7', True): {'prob_fold': 0.0, 'prob_call': 0.6687, 'prob_raise': 0.3313}, ('5', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.7168, 'prob_raise': 0.2832}, ('8', '9', False): {'prob_fold': 0.0, 'prob_call': 0.8431, 'prob_raise': 0.1569}, ('9', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.5604, 'prob_raise': 0.4396}, ('K', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.4162, 'prob_raise': 0.5838}, ('3', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.8943, 'prob_raise': 0.1057}, ('3', '6', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.8506, 'prob_raise': 0.1494}, ('8', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.7063, 'prob_raise': 0.2937}, ('8', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.7861, 'prob_raise': 0.2139}, ('2', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.6258, 'prob_raise': 0.3742}, ('5', '6', True): {'prob_fold': 0.0, 'prob_call': 0.6055, 'prob_raise': 0.3945}, ('6', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.948, 'prob_raise': 0.052}, ('3', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.6057, 'prob_raise': 0.3943}, ('8', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.4995, 'prob_raise': 0.5005}, ('2', '2', False): {'prob_fold': 0.0, 'prob_call': 0.9755, 'prob_raise': 0.0245}, ('2', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.9153, 'prob_raise': 0.0847}, ('T', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.5237, 'prob_raise': 0.4763}, ('3', '4', True): {'prob_fold': 0.0, 'prob_call': 0.9655, 'prob_raise': 0.0345}, ('2', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.2889, 'prob_raise': 0.7111}, ('4', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.5678, 'prob_raise': 0.4322}, ('2', '3', False): {'prob_fold': 0.0217, 'prob_call': 0.9783, 'prob_raise': 0.0}, ('4', '6', True): {'prob_fold': 0.0, 'prob_call': 0.7305, 'prob_raise': 0.2695}, ('5', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '3', False): {'prob_fold': 0.0, 'prob_call': 0.6561, 'prob_raise': 0.3439}, ('6', '6', False): {'prob_fold': 0.0, 'prob_call': 0.3658, 'prob_raise': 0.6342}, ('7', '9', True): {'prob_fold': 0.0, 'prob_call': 0.6367, 'prob_raise': 0.3633}, ('9', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.4723, 'prob_raise': 0.5277}, ('5', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.7298, 'prob_raise': 0.2702}, ('Q', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.4354, 'prob_raise': 0.5646}, ('T', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.5169, 'prob_raise': 0.4831}, ('3', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '8', True): {'prob_fold': 0.0, 'prob_call': 0.6398, 'prob_raise': 0.3602}, ('4', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '4', False): {'prob_fold': 0.0064, 'prob_call': 0.9936, 'prob_raise': 0.0}, ('T', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.5199, 'prob_raise': 0.4801}, ('6', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.8373, 'prob_raise': 0.1627}, ('4', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '7', False): {'prob_fold': 0.0, 'prob_call': 0.9333, 'prob_raise': 0.0667}, ('Q', 'K', False): {'prob_fold': 0.0, 'prob_call': 0.5364, 'prob_raise': 0.4636}, ('2', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'T', True): {'prob_fold': 0.0, 'prob_call': 0.6658, 'prob_raise': 0.3342}, ('7', '8', False): {'prob_fold': 0.0, 'prob_call': 0.8819, 'prob_raise': 0.1181}, ('5', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.6535, 'prob_raise': 0.3465}, ('3', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.9759, 'prob_raise': 0.0241}, ('3', '8', False): {'prob_fold': 0.0435, 'prob_call': 0.9565, 'prob_raise': 0.0}, ('5', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'T', False): {'prob_fold': 0.0, 'prob_call': 0.7312, 'prob_raise': 0.2688}, ('2', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.5921, 'prob_raise': 0.4079}, ('4', '9', False): {'prob_fold': 0.0034, 'prob_call': 0.9966, 'prob_raise': 0.0}, ('6', '9', True): {'prob_fold': 0.0, 'prob_call': 0.7175, 'prob_raise': 0.2825}, ('J', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.3621, 'prob_raise': 0.6379}, ('K', 'A', False): {'prob_fold': 0.0, 'prob_call': 0.4578, 'prob_raise': 0.5422}, ('6', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.8132, 'prob_raise': 0.1868}, ('3', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', '8', False): {'prob_fold': 0.0, 'prob_call': 0.2155, 'prob_raise': 0.7845}, ('7', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.6407, 'prob_raise': 0.3593}, ('2', '6', False): {'prob_fold': 0.0569, 'prob_call': 0.9431, 'prob_raise': 0.0}, ('5', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'Q', True): {'prob_fold': 0.0, 'prob_call': 0.5771, 'prob_raise': 0.4229}, ('J', 'K', True): {'prob_fold': 0.0, 'prob_call': 0.4982, 'prob_raise': 0.5018}, ('T', 'Q', False): {'prob_fold': 0.0, 'prob_call': 0.6339, 'prob_raise': 0.3661}, ('5', 'A', True): {'prob_fold': 0.0, 'prob_call': 0.5579, 'prob_raise': 0.4421}, ('8', 'J', False): {'prob_fold': 0.0, 'prob_call': 0.9498, 'prob_raise': 0.0502}, ('5', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '7', False): {'prob_fold': 0.0, 'prob_call': 0.2923, 'prob_raise': 0.7077}, ('2', '7', False): {'prob_fold': 0.2697, 'prob_call': 0.7303, 'prob_raise': 0.0}, ('6', 'J', True): {'prob_fold': 0.0, 'prob_call': 0.955, 'prob_raise': 0.045}, ('8', '9', True): {'prob_fold': 0.0, 'prob_call': 0.6119, 'prob_raise': 0.3881}}

call_raise_raise_raise_dict = {('9', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '5', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '5', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '4', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('A', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('K', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '4', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '3', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '4', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '6', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '5', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('K', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '6', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '6', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '2', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '4', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '3', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '6', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '3', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '8', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '4', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('Q', 'K', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '9', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('K', 'A', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'T', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '7', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', '8', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('4', '5', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('9', 'Q', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('J', 'K', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('T', 'Q', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'A', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', 'J', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('5', 'T', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('3', '6', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('7', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('2', '7', False): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('6', 'J', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}, ('8', '9', True): {'prob_fold': 0.0, 'prob_call': 1.0, 'prob_raise': 0.0}}

preflop_dict = {
  (): no_action_dict,
  ("raise",): raise_dict,
  ("raise", "raise"): raise_raise_dict,
  ("raise", "raise", "raise"): raise_raise_raise_dict,
  ("call",): call_dict,
  ("call", "raise"): call_raise_dict,
  ("call", "raise", "raise"): call_raise_raise_dict,
  ("call", "raise", "raise", "raise"): call_raise_raise_raise_dict
}
class SmartestPlayerImprovedPreflop(BasePokerPlayer):
  '''
  self:
  street
  small_blind_amount
  hole_card
  players
  player_pos
  pot
  community_card
  small_blind_pos
  big_blind_has_spoken
  street_in_str
  '''
  LOG = False
  MIN_VALUE = -999.0
  def __init__(self):
    self.oppo_num_raises = 0
    self.count = 0
    self.player_pos = None
    self.big_blind_has_spoken = False
    self.num_raises = 0
    self.small_blind_pos = 0
    self.updated_commitments = False
    self.commitments = [0,0]
    self.missed_uuid = None
    self.missed_action = None
    self.raise_amt = [20, 20, 40, 40]
    self.alpha = 0.8

    '''
    first index denotes the number of times the opponent has raised.
    second index denotes the frequency of fold, freq. of call, freq. of raise.
    '''
    self.raise_freq = np.array([
    [1.0,1.0,1.0],
    [1.0,1.0,1.0],
    [1.0,1.0,1.0],
    [1.0,1.0,1.0],
    [1.0,1.0,1.0],
    ])
    '''
    first index denotes the frequencies of the following win condition:
    0 -> HIGHCARD, high <= 7
    1 -> HIGHCARD, high > 7
    2 -> ONEPAIR, high <= 7
    3 -> ONEPAIR, high > 7
    4 -> TWOPAIR, high <= 7
    5 -> TWOPAIR, high > 7
    6 -> THREECARD, high <= 7
    7 -> THREECARD, high > 7
    8 -> STRAIGHT, high <= 7
    9 -> STRAIGHT, high > 7
    10 -> FLUSH, high <= 7
    11 -> FLUSH, high > 7
    12 -> FULLHOUSE, high <= 7
    13 -> FULLHOUSE, high > 7
    14 -> FOURCARD, high <= 7
    15 -> FOURCARD, high > 7
    16 -> STRAIGHTFLUSH, high <= 7
    17 -> STRAIGHTFLUSH, high > 7
    '''
    self.showdown_card_freq = np.array([1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    self.hand_strength_to_number = {
    'FLASH':5,
    'FOURCARD':7,
    'FULLHOUSE':6,
    'HIGHCARD':0,
    'ONEPAIR':1,
    'STRAIGHT':4,
    'STRAIGHTFLASH':8,
    'THREECARD':3,
    'TWOPAIR':2}


  def declare_action(self, valid_actions, hole_card, round_state):
    self.player_pos = round_state['next_player']

    if not self.updated_commitments:
      self.player_committed_amt = self.commitments[self.player_pos]
      self.oppo_committed_amt = self.commitments[abs(self.player_pos-1)]
      if self.missed_uuid == self.players[abs(self.player_pos-1)].uuid:
        self.raise_freq[0][self.missed_action] = self.raise_freq[0][self.missed_action] + 1
      self.updated_commitments = True

    if self.street == 0:
      if self.preflop_move(round_state) == "fold":
        return "fold"
      elif self.preflop_move(round_state) == "call":
        self.player_committed_amt = self.oppo_committed_amt
        return "call"
      elif self.preflop_move(round_state) == "raise":
        self.player_committed_amt = self.oppo_committed_amt + self.raise_amt[self.street]
        self.num_raises = self.num_raises + 1
        return "raise"
      
      # if len(valid_actions) == 3 and self.is_excellent_hole_card(round_state):
      #   self.player_committed_amt = self.oppo_committed_amt + self.raise_amt[self.street]
      #   self.num_raises = self.num_raises + 1
      #   return "raise"
      # elif self.is_decent_hole_card(round_state):
      #   self.player_committed_amt = self.oppo_committed_amt
      #   return "call"
      # else:
      #   return "fold"

    total_num_oppo_raise = self.compute_total_num_player_raise(self.players,abs(self.player_pos-1))
    total_num_player_raise = self.compute_total_num_player_raise(self.players,self.player_pos)
    num_oppo_raise = self.compute_num_player_raise(self.players,abs(self.player_pos-1))
    num_player_raise = self.compute_num_player_raise(self.players,self.player_pos)
    index = self.compute_action_for_player(self.player_pos,self.street,self.small_blind_pos, \
      self.player_committed_amt,self.oppo_committed_amt,self.num_raises,self.hole_card, \
      self.community_card,self.big_blind_has_spoken, total_num_player_raise, total_num_oppo_raise,\
      num_player_raise, num_oppo_raise)
    if index == 1:
      self.player_committed_amt = self.oppo_committed_amt
    elif index == 2:
      self.player_committed_amt = self.oppo_committed_amt+self.raise_amt[self.street]
      self.num_raises = self.num_raises + 1
    else:
      pass
    return valid_actions[index]["action"]

  def compute_action_for_player(self,player_pos,street,small_blind_pos, player_committed_amt, \
    oppo_committed_amt, total_num_raises, hole_card, community_card, big_blind_has_spoken, \
    total_num_player_raise, total_num_oppo_raise,num_player_raise, num_oppo_raise):
    if self.LOG:
      output_log_message_for_compute_action_for_player(player_pos,street,small_blind_pos,
        player_committed_amt, oppo_committed_amt, total_num_raises, hole_card, community_card,\
        players,big_blind_has_spoken, total_num_player_raise, total_num_oppo_raise, \
        num_player_raise, num_oppo_raise)

    eva = [self.MIN_VALUE,self.MIN_VALUE,self.MIN_VALUE] # fold, call ,raise

    # Fold Branch
    eva[0] = -1 * player_committed_amt

    # Call/Check Branch
    if player_pos == small_blind_pos:
      if self.big_blind_has_spoken:
        eva[1] = self.evaluate_chance_node(player_pos,street,small_blind_pos, oppo_committed_amt, \
          oppo_committed_amt, total_num_raises, hole_card, community_card, \
          big_blind_has_spoken,total_num_player_raise, total_num_oppo_raise,\
          num_player_raise, num_oppo_raise)
      else:
        eva[1] = self.evaluate_oppo_node(player_pos,street,small_blind_pos, oppo_committed_amt, \
          oppo_committed_amt, total_num_raises, hole_card, community_card, \
          big_blind_has_spoken,total_num_player_raise, total_num_oppo_raise, \
          num_player_raise, num_oppo_raise)
    else:
      # player is big blind
      eva[1] = self.evaluate_chance_node(player_pos,street,small_blind_pos,oppo_committed_amt, \
        oppo_committed_amt,total_num_raises, hole_card, community_card,True, \
        total_num_player_raise, total_num_oppo_raise,num_player_raise, num_oppo_raise)

    # Raise Branch
    if total_num_player_raise < 4 and num_oppo_raise+num_player_raise < 4:
      if player_pos == small_blind_pos:
        eva[2] = self.evaluate_player_node(player_pos,street,small_blind_pos, \
          oppo_committed_amt+self.raise_amt[street],oppo_committed_amt,total_num_raises+1, \
          hole_card,community_card,big_blind_has_spoken, total_num_player_raise+1, \
          total_num_oppo_raise,num_player_raise+1, num_oppo_raise)
      else:
        eva[2] = self.evaluate_player_node(player_pos,street,small_blind_pos, \
          oppo_committed_amt+self.raise_amt[street],oppo_committed_amt,total_num_raises+1, \
          hole_card,community_card,True, total_num_player_raise+1, total_num_oppo_raise,\
          num_player_raise+1, num_oppo_raise)

    return np.argmax(eva)

  def preflop_move(self, round_state):
    moves = ()
    moves_length = len(round_state["action_histories"]["preflop"]) - 2

    if moves_length > 0:
      for i in range(2, 2 + moves_length):
        next_move = round_state["action_histories"]["preflop"][i]["action"].lower()
        moves += (next_move,)
    print("CURRENT MOVES: ", moves)
    
    hc = []
    suits = []
    for c in self.hole_card:
      r = int(c.rank)
      hc.append(r)
      suits.append(c.suit)
    hc.sort()
    print("MY CARDS: ", hc, suits)
    first_num = c.RANK_MAP[hc[0]]
    second_num = c.RANK_MAP[hc[1]]
    same_suit = suits[0] == suits[1]
    tup = ()
    tup += (first_num,)
    tup += (second_num,)
    tup += (same_suit,)

    inner_dict = preflop_dict[moves]
    action_prob = inner_dict[tup]
    fold_prob = action_prob["prob_fold"]
    call_prob = action_prob["prob_call"]
    raise_prob = action_prob["prob_raise"]
    val = np.random.choice(np.arange(1, 4), p=[fold_prob, call_prob, raise_prob])
    if val == 1:
      return "fold"
    elif val == 2:
      return "call"
    else:
      return "raise"
  
  def is_decent_hole_card(self, round_state):
    moves = ()
    #print("CURRENT MOVES: ", round_state)
    moves_length = len(round_state["action_histories"]["preflop"]) - 2

    if moves_length > 0:
      for i in range(2, 2 + moves_length):
        next_move = round_state["action_histories"]["preflop"][i]["action"].lower()
        moves += (next_move,)


    hc = []
    suits = []
    for c in self.hole_card:
      r = int(c.rank)
      hc.append(r)
      suits.append(c.suit)
    hc.sort()
    first_num = c.RANK_MAP[hc[0]]
    second_num = c.RANK_MAP[hc[1]]
    same_suit = suits[0] == suits[1]
    tup = ()
    tup += (first_num,)
    tup += (second_num,)
    tup += (same_suit,)

    inner_dict = preflop_dict[moves]
    if tup in inner_dict:
      action = inner_dict[tup]
      if action == "call" or action == "raise":
        return True
      else:
        return False
    else:
      return False

  def is_excellent_hole_card(self, round_state):
    moves = ()
    #print("CURRENT MOVES: ", round_state)
    moves_length = len(round_state["action_histories"]["preflop"]) - 2

    if moves_length > 0:
      for i in range(2, 2 + moves_length):
        next_move = round_state["action_histories"]["preflop"][i]["action"].lower()
        moves += (next_move,)

    hc = []
    suits = []
    for c in self.hole_card:
      r = int(c.rank)
      hc.append(r)
      suits.append(c.suit)
    hc.sort()
    first_num = c.RANK_MAP[hc[0]]
    second_num = c.RANK_MAP[hc[1]]
    same_suit = suits[0] == suits[1]
    tup = ()
    tup += (first_num,)
    tup += (second_num,)
    tup += (same_suit,)

    inner_dict = preflop_dict[moves]
    if tup in inner_dict:
      action = inner_dict[tup]
      if action == "raise":
        return True
      else:
        return False
    else:
      return False

  def evaluate_player_node(self,player_pos,street,small_blind_pos, player_committed_amt, \
    oppo_committed_amt, total_num_raises, hole_card, community_card,big_blind_has_spoken, \
    total_num_player_raise, total_num_oppo_raise,num_player_raise, num_oppo_raise):
    if self.LOG:
      output_log_message_for_evaluate_player_node(player_pos,street,small_blind_pos,
        player_committed_amt, oppo_committed_amt, total_num_raises, hole_card, community_card, \
        big_blind_has_spoken, total_num_player_raise, total_num_oppo_raise,num_player_raise, \
        num_oppo_raise)

    if street == 4:
      if self.LOG:
        print("player 0")
      return self.evaluate_terminal_node(player_pos,street,small_blind_pos, player_committed_amt, \
        oppo_committed_amt, total_num_raises, hole_card, community_card,big_blind_has_spoken, \
        total_num_player_raise, total_num_oppo_raise,num_player_raise, num_oppo_raise)

    eva = [self.MIN_VALUE,self.MIN_VALUE,self.MIN_VALUE] # fold, call ,raise

    # Fold Branch
    eva[0] = -1 * player_committed_amt

    # Call/ Check Branch
    if player_pos == small_blind_pos:
      if big_blind_has_spoken:
        if self.LOG:
          print("player 1.1")
        eva[1] = self.evaluate_chance_node(player_pos,street,small_blind_pos, oppo_committed_amt, \
          oppo_committed_amt, total_num_raises, hole_card, community_card,\
          big_blind_has_spoken,total_num_player_raise, total_num_oppo_raise,num_player_raise, \
          num_oppo_raise)
      else:
        if self.LOG:
          print("player 1.2")
        eva[1] = self.evaluate_oppo_node(player_pos,street,small_blind_pos, oppo_committed_amt, \
          oppo_committed_amt, total_num_raises, hole_card, community_card, \
          big_blind_has_spoken,total_num_player_raise, total_num_oppo_raise,num_player_raise, \
          num_oppo_raise)
    else:
    # player is big blind
      if self.LOG:
        print("player 1.3")
      eva[1] = self.evaluate_chance_node(player_pos,street,small_blind_pos,oppo_committed_amt, \
        oppo_committed_amt,total_num_raises, hole_card, community_card,True, \
        total_num_player_raise, total_num_oppo_raise,num_player_raise, num_oppo_raise)

    # Raise Branch
    if total_num_player_raise < 4 and num_oppo_raise+num_player_raise < 4:
      if player_pos == small_blind_pos:
        if self.LOG:
          print("player 2.1")
        eva[2] = self.evaluate_oppo_node(player_pos,street,small_blind_pos, \
          oppo_committed_amt+self.raise_amt[street],oppo_committed_amt, total_num_raises+1, \
          hole_card, community_card,big_blind_has_spoken,total_num_player_raise+1, \
          total_num_oppo_raise,num_player_raise+1, num_oppo_raise)
      else:
        if self.LOG:
          print("player 2.2")
        eva[2] = self.evaluate_oppo_node(player_pos,street,small_blind_pos, \
          oppo_committed_amt+self.raise_amt[street],oppo_committed_amt, total_num_raises+1, \
          hole_card, community_card,True,total_num_player_raise+1, total_num_oppo_raise,\
          num_player_raise+1, num_oppo_raise)
    return max(eva)

  def evaluate_oppo_node(self,player_pos,street,small_blind_pos, player_committed_amt, \
    oppo_committed_amt, total_num_raises, hole_card, community_card, big_blind_has_spoken, \
    total_num_player_raise, total_num_oppo_raise,num_player_raise, num_oppo_raise):
    if self.LOG:
      output_log_message_for_evaluate_oppo_node(player_pos,street,small_blind_pos, player_committed_amt, \
    oppo_committed_amt, total_num_raises, hole_card, community_card,players,big_blind_has_spoken, \
    total_num_player_raise, total_num_oppo_raise,num_player_raise, num_oppo_raise)

    if street == 4:
      if self.LOG:
        print("oppo 0")
      return self.evaluate_terminal_node(player_pos,street,small_blind_pos, player_committed_amt, \
        oppo_committed_amt, total_num_raises, hole_card, community_card, big_blind_has_spoken, \
        total_num_player_raise, total_num_oppo_raise,num_player_raise, num_oppo_raise)

    s = sum(self.raise_freq[num_oppo_raise])
    fold_freq,call_freq,raise_freq=1/3,1/3,1/3
    if s != 0:
      fold_freq,call_freq,raise_freq = self.raise_freq[num_oppo_raise][0]/s, \
      self.raise_freq[num_oppo_raise][1]/s,self.raise_freq[num_oppo_raise][2]/s
    eva = [self.MIN_VALUE,self.MIN_VALUE,self.MIN_VALUE] # fold, call ,raise

    # Fold Branch
    eva[0] = oppo_committed_amt

    # Call/Check Branch
    if player_pos == small_blind_pos:
      if self.LOG:
        print("oppo 1.1")
      eva[1] = self.evaluate_chance_node(player_pos,street,small_blind_pos, player_committed_amt, \
        player_committed_amt, total_num_raises,hole_card,community_card,True, \
        total_num_player_raise, total_num_oppo_raise,num_player_raise, num_oppo_raise)
    else:
      # player is big blind
      if big_blind_has_spoken:
        if self.LOG:
          print("oppo 1.2")
        eva[1] = self.evaluate_chance_node(player_pos,street,small_blind_pos, player_committed_amt, \
          player_committed_amt, total_num_raises,hole_card,community_card,\
          big_blind_has_spoken, total_num_player_raise, total_num_oppo_raise,num_player_raise, \
          num_oppo_raise)
      else:
        if self.LOG:
          print("oppo 1.3")
        eva[1] = self.evaluate_player_node(player_pos,street,small_blind_pos, player_committed_amt, \
          player_committed_amt, total_num_raises,hole_card,community_card, \
          big_blind_has_spoken,total_num_player_raise, total_num_oppo_raise,num_player_raise, \
          num_oppo_raise)

    # Raise Branch
    if total_num_oppo_raise < 4 and num_oppo_raise+num_player_raise < 4:
        if player_pos == small_blind_pos:
          if self.LOG:
            print("oppo 2.1")
          eva[2] = self.evaluate_player_node(player_pos,street,small_blind_pos,player_committed_amt, \
            player_committed_amt+self.raise_amt[street],total_num_raises+1,hole_card,community_card, \
            True,total_num_player_raise, total_num_oppo_raise+1,num_player_raise, num_oppo_raise+1)
        else:
          if self.LOG:
            print("oppo 2.2")
          eva[2] = self.evaluate_player_node(player_pos,street,small_blind_pos,player_committed_amt, \
            player_committed_amt+self.raise_amt[street],total_num_raises+1,hole_card,community_card, \
            big_blind_has_spoken,total_num_player_raise, total_num_oppo_raise+1,\
            num_player_raise, num_oppo_raise+1)
    else:
      return eva[0] * fold_freq + eva[1] * call_freq
    return eva[0] * fold_freq + eva[1] * call_freq + eva[2] * raise_freq

  def evaluate_chance_node(self,player_pos,street,small_blind_pos, player_committed_amt, \
    oppo_committed_amt, total_num_raises, hole_card, community_card,big_blind_has_spoken, \
    total_num_player_raise, total_num_oppo_raise,num_player_raise, num_oppo_raise):
    if self.LOG:
      output_log_message_for_evaluate_chance_node(player_pos,street,small_blind_pos, \
        player_committed_amt, oppo_committed_amt, total_num_raises, hole_card, community_card, \
        big_blind_has_spoken, total_num_player_raise, total_num_oppo_raise,num_player_raise,\
         num_oppo_raise)

    if street == 4:
      return self.evaluate_terminal_node(player_pos,street,small_blind_pos, player_committed_amt, \
        oppo_committed_amt, total_num_raises, hole_card, community_card,big_blind_has_spoken, \
        total_num_player_raise, total_num_oppo_raise,num_player_raise, num_oppo_raise)

    eva = 0
    num_cards_left = 50-len(community_card)

    cards = [c.to_id() for c in community_card]
    for i in _pick_unused_card(1, hole_card+community_card):
      cards.append(i.to_id())
      cc = [Card.from_id(c) for c in cards]
      if player_pos == small_blind_pos:
        if self.LOG:
          print("chance 1.1")
        eva = self.evaluate_player_node(player_pos,street+1,small_blind_pos,player_committed_amt, \
          oppo_committed_amt,total_num_raises,hole_card,cc,False, total_num_player_raise, \
          total_num_oppo_raise,0,0) + eva
      else:
        if self.LOG:
          print("chance 1.2")
        eva = self.evaluate_oppo_node(player_pos,street+1,small_blind_pos,player_committed_amt, \
          oppo_committed_amt,total_num_raises,hole_card,cc,False,total_num_player_raise, \
          total_num_oppo_raise,0,0) + eva
    return eva/num_cards_left

  def evaluate_terminal_node(self,player_pos,street,small_blind_pos, player_committed_amt, \
    oppo_committed_amt, total_num_raises, hole_card, community_card,big_blind_has_spoken, \
    total_num_player_raise, total_num_oppo_raise,num_player_raise, num_oppo_raise):
    if self.LOG:
      output_log_message_for_evaluate_terminal_node(self,player_pos,street,small_blind_pos, \
        player_committed_amt, oppo_committed_amt, total_num_raises, hole_card, community_card, \
        big_blind_has_spoken, total_num_player_raise, total_num_oppo_raise,num_player_raise, \
        num_oppo_raise)
    s = sum(self.showdown_card_freq)
    index = self.compute_showdown_cards_index(self.hole_card)
    prob_win = 0
    for i in range(index):
      prob_win += self.showdown_card_freq[index]/s
    return prob_win * (player_committed_amt + oppo_committed_amt) - player_committed_amt

  def compute_total_num_player_raise(self,players,player_pos):
    raised_number = 0
    for action in players[player_pos].action_histories:
      if action["action"] == "RAISE":
        raised_number += 1
    histories = players[player_pos].round_action_histories
    for rounds in histories:
      if rounds == None:
        return raised_number
      else:
        for action in rounds:
          if action["action"] == "RAISE":
            raised_number += 1
    return raised_number

  def compute_num_player_raise(self,players,player_pos):
    raised_number = 0
    for action in players[player_pos].action_histories:
      if action["action"] == "RAISE":
        raised_number += 1
    return raised_number

  def receive_game_start_message(self, game_info):
    player0_uuid = game_info['seats'][0]['uuid']
    player0_stack = game_info['seats'][0]['stack']
    player0_name = game_info['seats'][0]['name']
    player1_uuid = game_info['seats'][1]['uuid']
    player1_stack = game_info['seats'][1]['stack']
    player1_name = game_info['seats'][1]['name']
    player0 = Player(player0_uuid,player0_stack,player0_name)
    player1 = Player(player1_uuid,player1_stack,player1_name)
    self.players = [player0,player1]
    self.oppo_committed_amt=0
    self.player_committed_amt=0
    self.num_raises=0
    '''
    pp = pprint.PrettyPrinter(indent=2)
    print("------------GAME_INFO/receive_game_start_message--------")
    pp.pprint(game_info)
    print("-------------------------------")
    '''

  def receive_round_start_message(self, round_count, hole_card, seats):
    self.hole_card = gen_cards(hole_card)
    '''
    pp = pprint.PrettyPrinter(indent=2)
    print("------------ROUND_COUNT/receive_round_start_message--------")
    pp.pprint(round_count)
    print("------------HOLE_CARD----------")
    pp.pprint(hole_card)
    print("------------SEATS----------")
    pp.pprint(seats)
    print("-------------------------------")
    '''

  def receive_street_start_message(self, street, round_state):
    '''
    pp = pprint.PrettyPrinter(indent=2)
    for p in self.players:
        pp.pprint(p.round_action_histories)
    print("------------STREET/receive_street_start_message--------")
    pp.pprint(street)
    print("------------ROUND_STATE----------")
    pp.pprint(round_state)
    print("-------------------------------")
    '''
    # https://stackoverflow.com/questions/9058305/getting-attributes-of-a-class
    all_attributes = inspect.getmembers(PokerConstants.Street, lambda a:not(inspect.isroutine(a)))
    attributes = [a for a in all_attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
    for a in attributes:
        if a[0] == street.upper():
            self.street = a[1]
    self.street_in_str = street
    self.pot = round_state['pot']['main']['amount']
    self.community_card = gen_cards(round_state['community_card'])
    self.small_blind_amount = round_state['small_blind_amount']
    self.small_blind_pos = round_state['small_blind_pos']
    if self.street == 0:
      self.oppo_num_raises = 0
      self.oppo_committed_amt=0
      self.player_committed_amt=0
      self.commitments[0]=0
      self.commitments[1]=0
      self.num_raises = 0
      self.raise_freq *= self.alpha
      self.showdown_card_freq *= self.alpha
      for p in self.players:
        p.clear_action_histories()

      self.players[round_state['big_blind_pos']].add_action_history(PokerConstants.Action.BIG_BLIND, \
        self.raise_amt[self.street], self.small_blind_amount, self.small_blind_amount)
      self.players[self.small_blind_pos].add_action_history(PokerConstants.Action.SMALL_BLIND, \
        self.small_blind_amount, self.small_blind_amount, self.small_blind_amount)
    else:
      for p in self.players:
        p.save_street_action_histories(self.street-1)
    self.big_blind_has_spoken = False

    if self.player_pos == None:
      if self.players[0].uuid == self.small_blind_pos:
          self.commitments[0] = self.small_blind_amount
          self.commitments[1] = self.small_blind_amount*2
      else:
        self.commitments[0] = self.small_blind_amount*2
        self.commitments[1] = self.small_blind_amount

    '''
    pp = pprint.PrettyPrinter(indent=2)
    for p in self.players:
        print("round_action_histories")
        pp.pprint(p.action_histories)
    print("------------STREET/receive_street_start_message--------")
    pp.pprint(street)
    print("------------ROUND_STATE----------")
    pp.pprint(round_state)
    print("-------------------------------")
    '''

  def receive_game_update_message(self, action, round_state):
    '''
    pp = pprint.PrettyPrinter(indent=2)
    print("------------ACTION/receive_game_update_message--------")
    pp.pprint(action)
    print("------------ROUND_STATE----------")
    pp.pprint(round_state)
    print("-------------------------------")
    '''
    p_action_in_str = action['action'].upper()
    p_action = None
    p_uuid = action['player_uuid']
    p_amount = action['amount']
    # https://stackoverflow.com/questions/9058305/getting-attributes-of-a-class
    all_attributes = inspect.getmembers(PokerConstants.Action, lambda a:not(inspect.isroutine(a)))
    attributes = [a for a in all_attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
    for a in attributes:
        if a[0] == p_action_in_str:
            p_action = a[1]
    for p in self.players:
        if p.uuid == p_uuid:
            p.add_action_history(p_action,p_amount)

    if not self.big_blind_has_spoken and p_uuid == round_state['big_blind_pos'] and \
    p_action != 'BIG_BLIND':
      self.big_blind_has_spoken = True

    if self.player_pos == None:
      self.missed_action = p_action
      self.missed_uuid = p_uuid
      if self.players[0].uuid == p_uuid:
        if p_action_in_str == "RAISE":
          self.commitments[0] = self.commitments[1] + 2*self.small_blind_amount
          self.num_raises = self.num_raises + 1
        elif p_action_in_str == "CALL":
          self.commitments[0] = self.commitments[1]
      else:
        if p_action_in_str == "RAISE":
          self.commitments[1] = self.commitments[0] + self.small_blind_amount*2
          self.num_raises = self.num_raises + 1
        elif p_action_in_str == "CALL":
          self.commitments[1] = self.commitments[0]

    if self.player_pos != None:
        if p_uuid == self.players[abs(self.player_pos-1)].uuid and p_action_in_str == "RAISE":
          self.oppo_committed_amt = self.player_committed_amt + 2 * self.small_blind_amount
          self.raise_freq[self.oppo_num_raises][2] += 1
          self.num_raises += 1

          self.oppo_num_raises += 1
        elif p_uuid == self.players[abs(self.player_pos-1)].uuid and p_action_in_str == "CALL":
          self.oppo_committed_amt = self.player_committed_amt

          self.raise_freq[self.oppo_num_raises][1] += 1
        elif p_uuid == self.players[abs(self.player_pos-1)].uuid and p_action_in_str == "FOLD":
          self.raise_freq[self.oppo_num_raises][0] += 1

    '''
    pp = pprint.PrettyPrinter(indent=2)
    print("player_pos:" + str(self.player_pos))
    print("oppo_committed_amt:" + str(self.oppo_committed_amt))
    print("player_committed_amt:" + str(self.player_committed_amt))
    print("big_blind_has_spoken:" + str(self.big_blind_has_spoken))
    print("num_raises:" + str(self.num_raises))
    print("small_blind_pos:" + str(self.small_blind_pos))
    for p in self.players:
      print("===================")
      pp.pprint(p.action_histories)
    '''

  def receive_round_result_message(self, winners, hand_info, round_state):
    if hand_info:
      oppo_hole_card = gen_cards(hand_info[abs(self.player_pos-1)]['hand']['card'])
      index = self.compute_showdown_cards_index(oppo_hole_card)
      self.showdown_card_freq[index] += 1

  def compute_showdown_cards_index(self, hole_card):
    hand_evaluator = HandEvaluator()
    b = bin(hand_evaluator.eval_hand(hole_card, self.community_card))
    hand_high_in_deci= int(b[2:-12],2)
    l = len(b) - 16
    index = 0
    if len(b) > 18:
      index = int(math.log(int(b[0:l],2),2)+1) * 2
      if hand_high_in_deci > 7:
        index += 1
    return index

    '''
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(self.showdown_card_freq)
    pp.pprint(self.raise_freq)

    for p in self.players:
        pp.pprint(p.round_action_histories)
    for p in self.players:
        pp.pprint(p.action_histories)
    print("------------WINNERS/receive_round_result_message--------")
    pp.pprint(winners)
    print("------------HAND_INFO----------")
    pp.pprint(hand_info)
    print("------------ROUND_STATE----------")
    pp.pprint(round_state)
    print("-------------------------------")
    '''

  def output_log_message_for_compute_action_for_player(self,player_pos,street,small_blind_pos, \
    player_committed_amt, oppo_committed_amt, total_num_raises, hole_card, community_card, \
    big_blind_has_spoken, num_player_raise, num_oppo_raise):
    print("compute_action_for_player ###########################################")
    pp = pprint.PrettyPrinter(indent=2)
    print("player_pos: "+ str(player_pos))
    print("street: "+ str(street))
    print("small_blind_pos: "+ str(small_blind_pos))
    print("player_committed_amt: "+ str(player_committed_amt))
    print("oppo_committed_amt: "+ str(oppo_committed_amt))
    print("total_num_raises: "+ str(total_num_raises))
    print("big_blind_has_spoken: "+ str(big_blind_has_spoken))

  def output_log_message_for_evaluate_player_node(self,player_pos,street,small_blind_pos, \
    player_committed_amt, oppo_committed_amt, total_num_raises, hole_card, community_card, \
    big_blind_has_spoken, num_player_raise, num_oppo_raise):
    pp = pprint.PrettyPrinter(indent=2)
    print("evaluate_player_node ###########################################")
    print("player_pos: "+ str(player_pos))
    print("street: "+ str(street))
    print("small_blind_pos: "+ str(small_blind_pos))
    print("player_committed_amt: "+ str(player_committed_amt))
    print("oppo_committed_amt: "+ str(oppo_committed_amt))
    print("total_num_raises: "+ str(total_num_raises))
    print("big_blind_has_spoken: "+ str(big_blind_has_spoken))
    print("oppo num raises: " + str(self.compute_num_player_raise_(players,abs(player_pos-1))))
    for p in players:
      pp.pprint(p.action_histories)
      pp.pprint(p.round_action_histories)

  def output_log_message_for_evaluate_oppo_node(self,player_pos,street,small_blind_pos, \
    player_committed_amt, oppo_committed_amt, total_num_raises, hole_card, community_card, \
    big_blind_has_spoken, num_player_raise, num_oppo_raise):
    pp = pprint.PrettyPrinter(indent=2)
    print("evaluate_oppo_node ###########################################")
    print("player_pos: "+ str(player_pos))
    print("street: "+ str(street))
    print("small_blind_pos: "+ str(small_blind_pos))
    print("player_committed_amt: "+ str(player_committed_amt))
    print("oppo_committed_amt: "+ str(oppo_committed_amt))
    print("total_num_raises: "+ str(total_num_raises))
    print("big_blind_has_spoken: "+ str(big_blind_has_spoken))
    print("oppo num raises: " + str(self.compute_num_player_raise(players,abs(player_pos-1))))
    for p in players:
      pp.pprint(p.action_histories)
      pp.pprint(p.round_action_histories)

  def output_log_message_for_evaluate_chance_node(self,player_pos,street,small_blind_pos, \
    player_committed_amt, oppo_committed_amt, total_num_raises, hole_card, community_card, \
    big_blind_has_spoken, num_player_raise, num_oppo_raise):
    pp = pprint.PrettyPrinter(indent=2)
    print("evaluate_chance_node ###########################################")
    print("player_pos: "+ str(player_pos))
    print("street: "+ str(street))
    print("small_blind_pos: "+ str(small_blind_pos))
    print("player_committed_amt: "+ str(player_committed_amt))
    print("oppo_committed_amt: "+ str(oppo_committed_amt))
    print("total_num_raises: "+ str(total_num_raises))
    print("big_blind_has_spoken: "+ str(big_blind_has_spoken))
    for p in players:
      pp.pprint(p.action_histories)

  def output_log_message_for_evaluate_terminal_node(self,player_pos,street,small_blind_pos, \
    player_committed_amt, oppo_committed_amt, total_num_raises, hole_card, community_card, \
    big_blind_has_spoken, num_player_raise, num_oppo_raise):
    print("terminate")

def setup_ai():

  return SmartestPlayer2()

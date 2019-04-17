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

class SmarterPlayerImprovedPreflop(BasePokerPlayer):
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
  
  #This is the preflop strategy that is referenced from University of Alberta Cepheus Poker Project.
  #It is a lookup table that determines the strength of the hand (and subsequently next move that the player
  #is going to make) based entirely on the 2 starting cards that the player has and the moves between the
  #two players in the preflop stage.
  #outer dictionary: key is the moves history, value is the inner dictionary for each moves history
  #inner dictionary: key is a tuple of (first_card, second_card, same_suit), value is the optimal action
  
  no_action_dict = {('2', '3', True): 'raise', ('2', '3', False): 'fold', ('2', '4', True): 'raise', ('2', '4', False): 'fold', ('3', '4', True): 'raise', ('3', '4', False): 'fold', ('2', '5', True): 'raise', ('2', '5', False): 'fold', ('3', '5', True): 'raise', ('3', '5', False): 'fold', ('4', '5', True): 'raise', ('4', '5', False): 'raise', ('2', '6', True): 'raise', ('2', '6', False): 'fold', ('3', '6', True): 'raise', ('3', '6', False): 'fold', ('4', '6', True): 'raise', ('4', '6', False): 'raise', ('5', '6', True): 'raise', ('5', '6', False): 'raise', ('2', '7', True): 'raise', ('2', '7', False): 'fold', ('3', '7', True): 'raise', ('3', '7', False): 'fold', ('4', '7', True): 'raise', ('4', '7', False): 'fold', ('5', '7', True): 'raise', ('5', '7', False): 'raise', ('6', '7', True): 'raise', ('6', '7', False): 'raise', ('2', '8', True): 'raise', ('2', '8', False): 'fold', ('3', '8', True): 'raise', ('3', '8', False): 'fold', ('4', '8', True): 'raise', ('4', '8', False): 'fold', ('5', '8', True): 'raise', ('5', '8', False): 'raise', ('6', '8', True): 'raise', ('6', '8', False): 'raise', ('7', '8', True): 'raise', ('7', '8', False): 'raise', ('2', '9', True): 'raise', ('2', '9', False): 'fold', ('3', '9', True): 'raise', ('3', '9', False): 'fold', ('4', '9', True): 'raise', ('4', '9', False): 'fold', ('5', '9', True): 'raise', ('5', '9', False): 'raise', ('6', '9', True): 'raise', ('6', '9', False): 'raise', ('7', '9', True): 'raise', ('7', '9', False): 'raise', ('8', '9', True): 'raise', ('8', '9', False): 'raise', ('2', 'T', True): 'raise', ('2', 'T', False): 'fold', ('3', 'T', True): 'raise', ('3', 'T', False): 'fold', ('4', 'T', True): 'raise', ('4', 'T', False): 'raise', ('5', 'T', True): 'raise', ('5', 'T', False): 'raise', ('6', 'T', True): 'raise', ('6', 'T', False): 'raise', ('7', 'T', True): 'raise', ('7', 'T', False): 'raise', ('8', 'T', True): 'raise', ('8', 'T', False): 'raise', ('9', 'T', True): 'raise', ('9', 'T', False): 'raise', ('2', 'J', True): 'raise', ('2', 'J', False): 'fold', ('3', 'J', True): 'raise', ('3', 'J', False): 'raise', ('4', 'J', True): 'raise', ('4', 'J', False): 'raise', ('5', 'J', True): 'raise', ('5', 'J', False): 'raise', ('6', 'J', True): 'raise', ('6', 'J', False): 'raise', ('7', 'J', True): 'raise', ('7', 'J', False): 'raise', ('8', 'J', True): 'raise', ('8', 'J', False): 'raise', ('9', 'J', True): 'raise', ('9', 'J', False): 'raise', ('T', 'J', True): 'raise', ('T', 'J', False): 'raise', ('2', 'Q', True): 'raise', ('2', 'Q', False): 'raise', ('3', 'Q', True): 'raise', ('3', 'Q', False): 'raise', ('4', 'Q', True): 'raise', ('4', 'Q', False): 'raise', ('5', 'Q', True): 'raise', ('5', 'Q', False): 'raise', ('6', 'Q', True): 'raise', ('6', 'Q', False): 'raise', ('7', 'Q', True): 'raise', ('7', 'Q', False): 'raise', ('8', 'Q', True): 'raise', ('8', 'Q', False): 'raise', ('9', 'Q', True): 'raise', ('9', 'Q', False): 'raise', ('T', 'Q', True): 'raise', ('T', 'Q', False): 'raise', ('J', 'Q', True): 'raise', ('J', 'Q', False): 'raise', ('2', 'K', True): 'raise', ('2', 'K', False): 'raise', ('3', 'K', True): 'raise', ('3', 'K', False): 'raise', ('4', 'K', True): 'raise', ('4', 'K', False): 'raise', ('5', 'K', True): 'raise', ('5', 'K', False): 'raise', ('6', 'K', True): 'raise', ('6', 'K', False): 'raise', ('7', 'K', True): 'raise', ('7', 'K', False): 'raise', ('8', 'K', True): 'raise', ('8', 'K', False): 'raise', ('9', 'K', True): 'raise', ('9', 'K', False): 'raise', ('T', 'K', True): 'raise', ('T', 'K', False): 'raise', ('J', 'K', True): 'raise', ('J', 'K', False): 'raise', ('Q', 'K', True): 'raise', ('Q', 'K', False): 'raise', ('2', 'A', True): 'raise', ('2', 'A', False): 'raise', ('3', 'A', True): 'raise', ('3', 'A', False): 'raise', ('4', 'A', True): 'raise', ('4', 'A', False): 'raise', ('5', 'A', True): 'raise', ('5', 'A', False): 'raise', ('6', 'A', True): 'raise', ('6', 'A', False): 'raise', ('7', 'A', True): 'raise', ('7', 'A', False): 'raise', ('8', 'A', True): 'raise', ('8', 'A', False): 'raise', ('9', 'A', True): 'raise', ('9', 'A', False): 'raise', ('T', 'A', True): 'raise', ('T', 'A', False): 'raise', ('J', 'A', True): 'raise', ('J', 'A', False): 'raise', ('Q', 'A', True): 'raise', ('Q', 'A', False): 'raise', ('K', 'A', True): 'raise', ('K', 'A', False): 'raise', ('2', '2', False): 'raise', ('3', '3', False): 'raise', ('4', '4', False): 'raise', ('5', '5', False): 'raise', ('6', '6', False): 'raise', ('7', '7', False): 'raise', ('8', '8', False): 'raise', ('9', '9', False): 'raise', ('T', 'T', False): 'raise', ('J', 'J', False): 'raise', ('Q', 'Q', False): 'raise', ('K', 'K', False): 'raise', ('A', 'A', False): 'raise'}

  raise_dict = {('2', '3', True): 'call', ('2', '3', False): 'fold', ('2', '4', True): 'call', ('2', '4', False): 'fold', ('3', '4', True): 'raise', ('3', '4', False): 'call', ('2', '5', True): 'call', ('2', '5', False): 'fold', ('3', '5', True): 'raise', ('3', '5', False): 'call', ('4', '5', True): 'raise', ('4', '5', False): 'call', ('2', '6', True): 'call', ('2', '6', False): 'fold', ('3', '6', True): 'call', ('3', '6', False): 'call', ('4', '6', True): 'raise', ('4', '6', False): 'call', ('5', '6', True): 'raise', ('5', '6', False): 'call', ('2', '7', True): 'call', ('2', '7', False): 'fold', ('3', '7', True): 'call', ('3', '7', False): 'fold', ('4', '7', True): 'call', ('4', '7', False): 'call', ('5', '7', True): 'raise', ('5', '7', False): 'call', ('6', '7', True): 'raise', ('6', '7', False): 'call', ('2', '8', True): 'call', ('2', '8', False): 'fold', ('3', '8', True): 'call', ('3', '8', False): 'fold', ('4', '8', True): 'call', ('4', '8', False): 'call', ('5', '8', True): 'raise', ('5', '8', False): 'call', ('6', '8', True): 'raise', ('6', '8', False): 'call', ('7', '8', True): 'raise', ('7', '8', False): 'call', ('2', '9', True): 'call', ('2', '9', False): 'call', ('3', '9', True): 'call', ('3', '9', False): 'call', ('4', '9', True): 'call', ('4', '9', False): 'call', ('5', '9', True): 'raise', ('5', '9', False): 'call', ('6', '9', True): 'raise', ('6', '9', False): 'call', ('7', '9', True): 'raise', ('7', '9', False): 'call', ('8', '9', True): 'raise', ('8', '9', False): 'call', ('2', 'T', True): 'call', ('2', 'T', False): 'call', ('3', 'T', True): 'call', ('3', 'T', False): 'call', ('4', 'T', True): 'call', ('4', 'T', False): 'call', ('5', 'T', True): 'call', ('5', 'T', False): 'call', ('6', 'T', True): 'raise', ('6', 'T', False): 'call', ('7', 'T', True): 'raise', ('7', 'T', False): 'call', ('8', 'T', True): 'raise', ('8', 'T', False): 'call', ('9', 'T', True): 'raise', ('9', 'T', False): 'call', ('2', 'J', True): 'call', ('2', 'J', False): 'call', ('3', 'J', True): 'call', ('3', 'J', False): 'call', ('4', 'J', True): 'raise', ('4', 'J', False): 'call', ('5', 'J', True): 'raise', ('5', 'J', False): 'call', ('6', 'J', True): 'call', ('6', 'J', False): 'call', ('7', 'J', True): 'raise', ('7', 'J', False): 'call', ('8', 'J', True): 'raise', ('8', 'J', False): 'call', ('9', 'J', True): 'raise', ('9', 'J', False): 'call', ('T', 'J', True): 'raise', ('T', 'J', False): 'call', ('2', 'Q', True): 'call', ('2', 'Q', False): 'call', ('3', 'Q', True): 'call', ('3', 'Q', False): 'call', ('4', 'Q', True): 'raise', ('4', 'Q', False): 'call', ('5', 'Q', True): 'raise', ('5', 'Q', False): 'call', ('6', 'Q', True): 'raise', ('6', 'Q', False): 'call', ('7', 'Q', True): 'raise', ('7', 'Q', False): 'call', ('8', 'Q', True): 'raise', ('8', 'Q', False): 'call', ('9', 'Q', True): 'raise', ('9', 'Q', False): 'call', ('T', 'Q', True): 'raise', ('T', 'Q', False): 'call', ('J', 'Q', True): 'raise', ('J', 'Q', False): 'call', ('2', 'K', True): 'call', ('2', 'K', False): 'call', ('3', 'K', True): 'call', ('3', 'K', False): 'call', ('4', 'K', True): 'raise', ('4', 'K', False): 'call', ('5', 'K', True): 'raise', ('5', 'K', False): 'call', ('6', 'K', True): 'raise', ('6', 'K', False): 'call', ('7', 'K', True): 'raise', ('7', 'K', False): 'call', ('8', 'K', True): 'raise', ('8', 'K', False): 'call', ('9', 'K', True): 'raise', ('9', 'K', False): 'call', ('T', 'K', True): 'raise', ('T', 'K', False): 'call', ('J', 'K', True): 'raise', ('J', 'K', False): 'raise', ('Q', 'K', True): 'raise', ('Q', 'K', False): 'raise', ('2', 'A', True): 'call', ('2', 'A', False): 'call', ('3', 'A', True): 'call', ('3', 'A', False): 'call', ('4', 'A', True): 'raise', ('4', 'A', False): 'call', ('5', 'A', True): 'raise', ('5', 'A', False): 'call', ('6', 'A', True): 'raise', ('6', 'A', False): 'call', ('7', 'A', True): 'raise', ('7', 'A', False): 'call', ('8', 'A', True): 'raise', ('8', 'A', False): 'raise', ('9', 'A', True): 'raise', ('9', 'A', False): 'raise', ('T', 'A', True): 'raise', ('T', 'A', False): 'raise', ('J', 'A', True): 'raise', ('J', 'A', False): 'raise', ('Q', 'A', True): 'raise', ('Q', 'A', False): 'raise', ('K', 'A', True): 'raise', ('K', 'A', False): 'raise', ('2', '2', False): 'call', ('3', '3', False): 'raise', ('4', '4', False): 'raise', ('5', '5', False): 'raise', ('6', '6', False): 'raise', ('7', '7', False): 'raise', ('8', '8', False): 'raise', ('9', '9', False): 'raise', ('T', 'T', False): 'raise', ('J', 'J', False): 'raise', ('Q', 'Q', False): 'raise', ('K', 'K', False): 'raise', ('A', 'A', False): 'raise'}

  raise__raise_dict = {('2', '3', True): 'call', ('2', '3', False): 'call', ('2', '4', True): 'call', ('2', '4', False): 'call', ('3', '4', True): 'call', ('3', '4', False): 'call', ('2', '5', True): 'call', ('2', '5', False): 'call', ('3', '5', True): 'call', ('3', '5', False): 'call', ('4', '5', True): 'call', ('4', '5', False): 'call', ('2', '6', True): 'call', ('2', '6', False): 'call', ('3', '6', True): 'call', ('3', '6', False): 'call', ('4', '6', True): 'call', ('4', '6', False): 'call', ('5', '6', True): 'call', ('5', '6', False): 'call', ('2', '7', True): 'call', ('2', '7', False): 'call', ('3', '7', True): 'call', ('3', '7', False): 'call', ('4', '7', True): 'call', ('4', '7', False): 'call', ('5', '7', True): 'call', ('5', '7', False): 'call', ('6', '7', True): 'call', ('6', '7', False): 'call', ('2', '8', True): 'call', ('2', '8', False): 'call', ('3', '8', True): 'call', ('3', '8', False): 'call', ('4', '8', True): 'call', ('4', '8', False): 'call', ('5', '8', True): 'call', ('5', '8', False): 'call', ('6', '8', True): 'call', ('6', '8', False): 'call', ('7', '8', True): 'call', ('7', '8', False): 'call', ('2', '9', True): 'call', ('2', '9', False): 'call', ('3', '9', True): 'call', ('3', '9', False): 'call', ('4', '9', True): 'call', ('4', '9', False): 'call', ('5', '9', True): 'call', ('5', '9', False): 'call', ('6', '9', True): 'call', ('6', '9', False): 'call', ('7', '9', True): 'call', ('7', '9', False): 'call', ('8', '9', True): 'call', ('8', '9', False): 'call', ('2', 'T', True): 'call', ('2', 'T', False): 'call', ('3', 'T', True): 'call', ('3', 'T', False): 'call', ('4', 'T', True): 'call', ('4', 'T', False): 'call', ('5', 'T', True): 'call', ('5', 'T', False): 'call', ('6', 'T', True): 'call', ('6', 'T', False): 'call', ('7', 'T', True): 'call', ('7', 'T', False): 'call', ('8', 'T', True): 'call', ('8', 'T', False): 'call', ('9', 'T', True): 'call', ('9', 'T', False): 'call', ('2', 'J', True): 'call', ('2', 'J', False): 'call', ('3', 'J', True): 'call', ('3', 'J', False): 'call', ('4', 'J', True): 'call', ('4', 'J', False): 'call', ('5', 'J', True): 'call', ('5', 'J', False): 'call', ('6', 'J', True): 'call', ('6', 'J', False): 'call', ('7', 'J', True): 'call', ('7', 'J', False): 'call', ('8', 'J', True): 'call', ('8', 'J', False): 'call', ('9', 'J', True): 'call', ('9', 'J', False): 'call', ('T', 'J', True): 'call', ('T', 'J', False): 'call', ('2', 'Q', True): 'call', ('2', 'Q', False): 'call', ('3', 'Q', True): 'call', ('3', 'Q', False): 'call', ('4', 'Q', True): 'call', ('4', 'Q', False): 'call', ('5', 'Q', True): 'call', ('5', 'Q', False): 'call', ('6', 'Q', True): 'call', ('6', 'Q', False): 'call', ('7', 'Q', True): 'call', ('7', 'Q', False): 'call', ('8', 'Q', True): 'call', ('8', 'Q', False): 'call', ('9', 'Q', True): 'call', ('9', 'Q', False): 'call', ('T', 'Q', True): 'call', ('T', 'Q', False): 'call', ('J', 'Q', True): 'call', ('J', 'Q', False): 'call', ('2', 'K', True): 'call', ('2', 'K', False): 'call', ('3', 'K', True): 'call', ('3', 'K', False): 'call', ('4', 'K', True): 'call', ('4', 'K', False): 'call', ('5', 'K', True): 'call', ('5', 'K', False): 'call', ('6', 'K', True): 'call', ('6', 'K', False): 'call', ('7', 'K', True): 'call', ('7', 'K', False): 'call', ('8', 'K', True): 'call', ('8', 'K', False): 'call', ('9', 'K', True): 'call', ('9', 'K', False): 'call', ('T', 'K', True): 'call', ('T', 'K', False): 'call', ('J', 'K', True): 'call', ('J', 'K', False): 'call', ('Q', 'K', True): 'call', ('Q', 'K', False): 'call', ('2', 'A', True): 'call', ('2', 'A', False): 'call', ('3', 'A', True): 'call', ('3', 'A', False): 'call', ('4', 'A', True): 'call', ('4', 'A', False): 'call', ('5', 'A', True): 'call', ('5', 'A', False): 'call', ('6', 'A', True): 'call', ('6', 'A', False): 'call', ('7', 'A', True): 'call', ('7', 'A', False): 'call', ('8', 'A', True): 'call', ('8', 'A', False): 'call', ('9', 'A', True): 'call', ('9', 'A', False): 'call', ('T', 'A', True): 'call', ('T', 'A', False): 'call', ('J', 'A', True): 'call', ('J', 'A', False): 'call', ('Q', 'A', True): 'call', ('Q', 'A', False): 'call', ('K', 'A', True): 'call', ('K', 'A', False): 'call', ('2', '2', False): 'call', ('3', '3', False): 'call', ('4', '4', False): 'call', ('5', '5', False): 'call', ('6', '6', False): 'call', ('7', '7', False): 'call', ('8', '8', False): 'call', ('9', '9', False): 'call', ('T', 'T', False): 'call', ('J', 'J', False): 'call', ('Q', 'Q', False): 'call', ('K', 'K', False): 'call', ('A', 'A', False): 'call'}

  raise__raise_raise_dict = {('2', '3', True): 'call', ('2', '3', False): 'call', ('2', '4', True): 'call', ('2', '4', False): 'call', ('3', '4', True): 'call', ('3', '4', False): 'call', ('2', '5', True): 'call', ('2', '5', False): 'call', ('3', '5', True): 'call', ('3', '5', False): 'call', ('4', '5', True): 'call', ('4', '5', False): 'call', ('2', '6', True): 'call', ('2', '6', False): 'call', ('3', '6', True): 'call', ('3', '6', False): 'call', ('4', '6', True): 'call', ('4', '6', False): 'call', ('5', '6', True): 'call', ('5', '6', False): 'call', ('2', '7', True): 'call', ('2', '7', False): 'call', ('3', '7', True): 'call', ('3', '7', False): 'call', ('4', '7', True): 'call', ('4', '7', False): 'call', ('5', '7', True): 'call', ('5', '7', False): 'call', ('6', '7', True): 'call', ('6', '7', False): 'call', ('2', '8', True): 'call', ('2', '8', False): 'call', ('3', '8', True): 'call', ('3', '8', False): 'call', ('4', '8', True): 'call', ('4', '8', False): 'call', ('5', '8', True): 'call', ('5', '8', False): 'call', ('6', '8', True): 'call', ('6', '8', False): 'call', ('7', '8', True): 'call', ('7', '8', False): 'call', ('2', '9', True): 'call', ('2', '9', False): 'call', ('3', '9', True): 'call', ('3', '9', False): 'call', ('4', '9', True): 'call', ('4', '9', False): 'call', ('5', '9', True): 'call', ('5', '9', False): 'call', ('6', '9', True): 'call', ('6', '9', False): 'call', ('7', '9', True): 'call', ('7', '9', False): 'call', ('8', '9', True): 'call', ('8', '9', False): 'call', ('2', 'T', True): 'call', ('2', 'T', False): 'call', ('3', 'T', True): 'call', ('3', 'T', False): 'call', ('4', 'T', True): 'call', ('4', 'T', False): 'call', ('5', 'T', True): 'call', ('5', 'T', False): 'call', ('6', 'T', True): 'call', ('6', 'T', False): 'call', ('7', 'T', True): 'call', ('7', 'T', False): 'call', ('8', 'T', True): 'call', ('8', 'T', False): 'call', ('9', 'T', True): 'call', ('9', 'T', False): 'call', ('2', 'J', True): 'call', ('2', 'J', False): 'call', ('3', 'J', True): 'call', ('3', 'J', False): 'call', ('4', 'J', True): 'call', ('4', 'J', False): 'call', ('5', 'J', True): 'call', ('5', 'J', False): 'call', ('6', 'J', True): 'call', ('6', 'J', False): 'call', ('7', 'J', True): 'call', ('7', 'J', False): 'call', ('8', 'J', True): 'call', ('8', 'J', False): 'call', ('9', 'J', True): 'call', ('9', 'J', False): 'call', ('T', 'J', True): 'call', ('T', 'J', False): 'call', ('2', 'Q', True): 'call', ('2', 'Q', False): 'call', ('3', 'Q', True): 'call', ('3', 'Q', False): 'call', ('4', 'Q', True): 'call', ('4', 'Q', False): 'call', ('5', 'Q', True): 'call', ('5', 'Q', False): 'call', ('6', 'Q', True): 'call', ('6', 'Q', False): 'call', ('7', 'Q', True): 'call', ('7', 'Q', False): 'call', ('8', 'Q', True): 'call', ('8', 'Q', False): 'call', ('9', 'Q', True): 'call', ('9', 'Q', False): 'call', ('T', 'Q', True): 'call', ('T', 'Q', False): 'call', ('J', 'Q', True): 'call', ('J', 'Q', False): 'call', ('2', 'K', True): 'call', ('2', 'K', False): 'call', ('3', 'K', True): 'call', ('3', 'K', False): 'call', ('4', 'K', True): 'call', ('4', 'K', False): 'call', ('5', 'K', True): 'call', ('5', 'K', False): 'call', ('6', 'K', True): 'call', ('6', 'K', False): 'call', ('7', 'K', True): 'call', ('7', 'K', False): 'call', ('8', 'K', True): 'call', ('8', 'K', False): 'call', ('9', 'K', True): 'call', ('9', 'K', False): 'call', ('T', 'K', True): 'call', ('T', 'K', False): 'call', ('J', 'K', True): 'call', ('J', 'K', False): 'call', ('Q', 'K', True): 'call', ('Q', 'K', False): 'call', ('2', 'A', True): 'call', ('2', 'A', False): 'call', ('3', 'A', True): 'call', ('3', 'A', False): 'call', ('4', 'A', True): 'call', ('4', 'A', False): 'call', ('5', 'A', True): 'call', ('5', 'A', False): 'call', ('6', 'A', True): 'call', ('6', 'A', False): 'call', ('7', 'A', True): 'call', ('7', 'A', False): 'call', ('8', 'A', True): 'call', ('8', 'A', False): 'call', ('9', 'A', True): 'call', ('9', 'A', False): 'call', ('T', 'A', True): 'call', ('T', 'A', False): 'call', ('J', 'A', True): 'call', ('J', 'A', False): 'call', ('Q', 'A', True): 'call', ('Q', 'A', False): 'call', ('K', 'A', True): 'call', ('K', 'A', False): 'call', ('2', '2', False): 'call', ('3', '3', False): 'call', ('4', '4', False): 'call', ('5', '5', False): 'call', ('6', '6', False): 'call', ('7', '7', False): 'call', ('8', '8', False): 'call', ('9', '9', False): 'call', ('T', 'T', False): 'call', ('J', 'J', False): 'call', ('Q', 'Q', False): 'call', ('K', 'K', False): 'call', ('A', 'A', False): 'call'}

  preflop_dict = {
    (): no_action_dict,
    ("raise"): raise_dict,
    ("raise", "raise"): raise__raise_dict,
    ("raise", "raise", "raise"): raise__raise_raise_dict
  }
  
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

    '''
    first index denotes the total number of (combined) raises.
    second index denotes the frequency of fold, freq. of call, freq. of raise.
    '''
    self.raise_freq = [
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0]
    ]
    '''
    first index denotes the frequencies of the following win condition:
    0 -> HIGHCARD
    1 -> ONEPAIR FOURCARD
    2 -> TWOPAIR
    3 -> THREECARD
    4 -> STRAIGHT
    5 -> FLUSH
    6 -> FULLHOUSE
    7 -> FOURCARD
    8 -> STRAIGHTFLUSH
    '''
    self.showdown_card_freq = [0,0,0,0,0,0,0,0,0]

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


      if len(valid_actions) == 3 and self.is_excellent_hole_card():
        self.player_committed_amt = self.oppo_committed_amt + self.raise_amt[self.street]
        self.num_raises = self.num_raises + 1
        return "raise"
      elif self.is_decent_hole_card():
        self.player_committed_amt = self.oppo_committed_amt
        return "call"
      else:
        return "fold"

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

  def is_decent_hole_card(self):
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
    tup += first_num
    tup += second_num
    tup += same_suit

    # if hc[0] == hc[1]:
    #   return True
    # elif hc[0] > 9 or hc[1] > 9:
    #   return True
    # else:
    #   return False

  def is_excellent_hole_card(self):
    hc = [int(c.rank) for c in self.hole_card]
    if hc[0] == hc[1]:
      return True
    elif hc[0] + hc[1] > 19:
      return True
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

    s = sum(self.raise_freq[total_num_raises])
    fold_freq,call_freq,raise_freq=1/3,1/3,1/3
    if s != 0:
      fold_freq,call_freq,raise_freq = self.raise_freq[total_num_raises][0]/s, \
      self.raise_freq[total_num_raises][1]/s,self.raise_freq[total_num_raises][2]/s
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
    hand_strength_index = self.hand_strength_to_number[HandEvaluator.gen_hand_rank_info(self.hole_card, \
      self.community_card)['hand']['strength']]
    prob_win = 0.5
    if s != 0:
      for i in range(hand_strength_index+1):
        prob_win = prob_win + self.showdown_card_freq[hand_strength_index]/s
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
      self.oppo_committed_amt=0
      self.player_committed_amt=0
      self.commitments[0]=0
      self.commitments[1]=0
      self.num_raises = 0
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
          self.raise_freq[self.num_raises][2] = self.raise_freq[self.num_raises][2] + 1
          self.num_raises = self.num_raises+1
        elif p_uuid == self.players[abs(self.player_pos-1)].uuid and p_action_in_str == "CALL":
          self.oppo_committed_amt = self.player_committed_amt
          self.raise_freq[self.num_raises][1] = self.raise_freq[self.num_raises][1] + 1
        elif p_uuid == self.players[abs(self.player_pos-1)].uuid and p_action_in_str == "FOLD":
          self.raise_freq[self.num_raises][0] = self.raise_freq[self.num_raises][0] + 1
        else:
          pass
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
      hs = hand_info[abs(self.player_pos-1)]['hand']['hand']['strength']
      hs_index = self.hand_strength_to_number[hs]
      self.showdown_card_freq[hs_index] = self.showdown_card_freq[hs_index] + 1

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
  return SmarterPlayer()

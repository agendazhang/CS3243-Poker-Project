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

class SmartestPlayer(BasePokerPlayer):
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
    hc = [int(c.rank) for c in self.hole_card]
    if hc[0] == hc[1]:
      return True
    elif hc[0] > 9 or hc[1] > 9:
      return True
    else:
      return False

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
      hs = hand_info[abs(self.player_pos-1)]['hand']['hand']['strength']
      hs_index = self.hand_strength_to_number[hs]
      self.showdown_card_freq[hs_index] = self.showdown_card_freq[hs_index] + 1
  
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
  
  return SmartestPlayer()
  
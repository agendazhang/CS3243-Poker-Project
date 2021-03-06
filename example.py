from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from smarterplayer import SmarterPlayer
from smarterplayerimprovedpreflop import Group40Player
from smartestplayer import SmartestPlayer
#from smartestplayer2 import SmartestPlayer2
from smartestplayerimprovedpreflop import SmartestPlayerImprovedPreflop

#TODO:config the config as our wish
config = setup_config(max_round=500, initial_stack=10000, small_blind_amount=10)



config.register_player(name="p1", algorithm=SmartestPlayer())
config.register_player(name="P2", algorithm=SmartestPlayerImprovedPreflop())


game_result = start_poker(config, verbose=1)

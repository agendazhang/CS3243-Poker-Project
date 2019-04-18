from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from smarterplayer import SmarterPlayer
from smarterplayerimprovedpreflop import Group40Player
from smartestplayer import SmartestPlayer
from smartestplayer2 import SmartestPlayer2

#TODO:config the config as our wish
config = setup_config(max_round=500, initial_stack=10000, small_blind_amount=10)



config.register_player(name="f1", algorithm=SmarterPlayer())
config.register_player(name="FT2", algorithm=SmartestPlayer2())


game_result = start_poker(config, verbose=1)

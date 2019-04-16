import json

class Triple:
    def __init__(self, first_num, second_num, same_suit):
        self.first_num = first_num
        self.second_num = second_num
        self.same_suit = same_suit

    def __str__(self):
        tup = []
        tup.append(self.first_num)
        tup.append(self.second_num)
        tup.append(self.same_suit)
        return tup

with open('rrrPreflop.json') as json_file:
    data = json.load(json_file)
    pdict = {}
    for p in data['data']:
        cards = str(p[u'cards'])
        first_num = cards[0]
        second_num = cards[2]
        first_suit = cards[1]
        second_suit = cards[3]
        same_suit = (first_suit == second_suit)
        
        action = ""

        prob_fold = float(p[u'fold'])
        prob_call = float(p[u'call'])
        prob_raise = float(p[u'fold'])
        

        prob_list = []
        prob_list.append(prob_fold)
        prob_list.append(prob_call)
        prob_list.append(prob_raise)
        prob_list.sort()

        if prob_list[2] == prob_fold:
            action = "fold"
        elif prob_list[2] == prob_call:
            action = "call"
        elif prob_list[2] == prob_raise:
            action = "raise"

        tup = ()
        tup += (first_num,)
        tup += (second_num,)
        tup += (same_suit,)
        
        #tup = Triple(first_num, second_num, same_suit)
        pdict[tup] = action

    print(pdict)


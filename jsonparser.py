import json

class Triple:
    def __init__(self, first_num, second_num, same_suit,
    prob_fold, prob_call, prob_raise):
        self.first_num = first_num
        self.second_num = second_num
        self.same_suit = same_suit
        #self.prob_fold = prob_fold
        #self.prob_call = prob_call
        #self.prob_raise = prob_raise

    def __str__(self):
        tup = []
        tup.append(self.first_num)
        tup.append(self.second_num)
        tup.append(self.same_suit)
        #tup.append(self.prob_fold)
        #tup.append(self.prob_call)
        #tup.append(self.prob_raise)
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

        action = {}

        prob_fold = float(p[u'fold'])
        prob_call = float(p[u'call'])
        prob_raise = float(p[u'raise'])

        action["prob_fold"] = prob_fold
        action["prob_call"] = prob_call
        action["prob_raise"] = prob_raise

        tup = ()
        tup += (first_num,)
        tup += (second_num,)
        tup += (same_suit,)
        #tup += (prob_fold, )
        #tup += (prob_call, )
        #tup += (prob_raise, )

        #tup = Triple(first_num, second_num, same_suit)
        pdict[tup] = action

    print(pdict)

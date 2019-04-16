import json
with open('initialPreflop.json') as json_file:
    data = json.load(json_file)
    for p in data['data']:
        print(p)

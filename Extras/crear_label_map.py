labels = [{'name':'A', 'id':1}, 
          {'name':'B', 'id':2},
          {'name':'C', 'id':3},
          {'name':'D', 'id':4},
        ]

with open('TLSA/Tensorflow/workspace/annotations/label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')
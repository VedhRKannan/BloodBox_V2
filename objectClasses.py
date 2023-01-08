import json

with open('objectOutput.json', 'r') as f:
    data = json.load(f)

classes = []

for obj in data:
    if obj == 'predictions':
        for pred in data[obj]:
            classes.append(pred['class'])

totals = []

for c in classes:
    count = classes.count(c)
    totals.append((c, count))

wbc = [t for t in totals if t[0] == 'WBC'][0][1]
platelets = [t for t in totals if t[0] == 'Platelets'][0][1]
rbc = [t for t in totals if t[0] == 'RBC'][0][1]

print('wbc:', wbc)
print('platelets:', platelets)
print('rbc:', rbc)

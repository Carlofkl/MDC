import os
import csv

folder = 'datasets/caltech101/101_ObjectCategories'

file_names = []
for file_name in os.listdir(folder):
    if file_name == 'BACKGROUND_Google':
        continue
    file_names.append(file_name)

file_names = sorted(file_names)
prompt = [['prompt', 'classname', 'classidx']]
prompots = [[f'a photo of a {x}.', x, idx] for idx, x in enumerate(file_names)]
prompt.extend(prompots)
print(prompt)

with open('prompts/caltech101_prompts.csv', 'w', newline='') as ff:
    writer = csv.writer(ff)
    writer.writerows(prompt)
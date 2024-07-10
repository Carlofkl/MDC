import os
import csv
import pickle

# with open('datasets/cifar-100-python/meta', 'rb') as f:
#     dict = pickle.load(f, encoding='bytes')
# print(dict)
# exit()

cls_file = 'prompts/cifar100_cls.txt'
with open(cls_file, 'r') as f:
    content = f.readlines()

content = sorted(content)

cls_name = [x.strip() for x in content]
prompt = [['prompt', 'classname', 'classidx']]
# print(cls_name)
cifar100_prompt = [[f'a blurry photo of a {x}.', x, idx] for idx, x in enumerate(cls_name)]
prompt.extend(cifar100_prompt)

with open('prompts/cifar100_prompts.csv', 'w', newline='') as ff:
    writer = csv.writer(ff)
    writer.writerows(prompt)

# print(prompt)
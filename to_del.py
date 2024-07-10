import torch

l = [1, 2, 3]
a = {str(k): {} for k in set(l)}
print(a)
for i in range(3):
    idx = i + 1
    for j in range(2):
        a[str(idx)].update(
            {str(j): torch.randn(2,2)}
        )

for k, v in a.items():
    for kk, vv in v.items():
        print(vv)
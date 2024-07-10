import argparse
import os
import os.path as osp
import torch
from tqdm import tqdm


def mean_per_class_acc(correct, labels):
    total_acc = 0
    for cls in torch.unique(labels):
        mask = labels == cls
        total_acc += correct[mask].sum() / mask.sum()
    return total_acc / len(torch.unique(labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str)
    parser.add_argument('--num', type=int, default=0)
    args = parser.parse_args()

    # get list of files
    files = os.listdir(args.folder)
    files = sorted([f for f in files if f.endswith('.pt')][:args.num]) if args.num > 0 else sorted([f for f in files if f.endswith('.pt')])

    preds = []
    labels = []
    res = {}
    wrong_label = {}
    for f in tqdm(files):
        data = torch.load(osp.join(args.folder, f))
        preds.append(data['pred'])
        labels.append(data['label'])
        
        l = data['label']
        if l not in res:
            res[l] = [0, 0]
            wrong_label[l] = []
        
        if data['pred'] == l:
            res[l][0] += 1
        else:
            res[l][1] += 1
            wrong_label[l].append(data['pred'])
            
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    # top 1
    correct = (preds == labels).sum().item()
    print(f'Top 1 acc: {correct / len(preds) * 100:.2f}%')
    # mean per class
    print(f'Mean per class acc: {mean_per_class_acc(preds == labels, labels) * 100:.2f}%')
    print('Acc pre class:')
    res = dict(sorted(res.items(), key=lambda x: x[0]))
    for k, v in res.items():
        print('{}: right={} wrong={}, acc={:.2f}\t'.format(k, v[0], v[1], v[0]/(v[1]+v[0])), wrong_label[k])


if __name__ == '__main__':
    main()

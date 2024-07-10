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
    parser.add_argument('--folder1', type=str)
    parser.add_argument('--folder2', type=str)
    parser.add_argument('--num', type=int, default=0)
    args = parser.parse_args()

    # get list of files
    Res = {}
    for idx, folder in enumerate([args.folder1, args.folder2]):
        files = os.listdir(folder)
        files = sorted([f for f in files if f.endswith('.pt')][:args.num]) if args.num > 0 else sorted([f for f in files if f.endswith('.pt')])

        preds = []
        labels = []
        res = {}
        for f in tqdm(files):
            data = torch.load(osp.join(folder, f))
            preds.append(data['pred'])
            labels.append(data['label'])
            
            l = data['label']
            if l not in res:
                res[l] = [0, 0]
            
            if data['pred'] == l:
                res[l][0] += 1
            else:
                res[l][1] += 1
                
        preds = torch.tensor(preds)
        labels = torch.tensor(labels)
        # top 1
        correct = (preds == labels).sum().item()
        print(f'Top 1 acc: {correct / len(preds) * 100:.2f}%')
        # mean per class
        print(f'Mean per class acc: {mean_per_class_acc(preds == labels, labels) * 100:.2f}%')
        print('Acc pre class:')
        res = dict(sorted(res.items(), key=lambda x: x[0]))
        Res[f'exp{idx+1}'] = res
        print('\n')
        # for k, v in res.items():
        #     print('{}: r={}, w={}, acc={:.2f}'.format(k, v[0], v[1], v[0]/(v[1]+v[0])))

    compare_res = {}
    for k, v in res.items():
        exp1_acc = Res['exp1'][k][0] * 100 / sum(Res['exp1'][k])
        exp2_acc = Res['exp2'][k][0] * 100 / sum(Res['exp2'][k])
        increase = exp2_acc - exp1_acc
        compare_res[k] = [f'{increase:.2f}', f'{exp1_acc:.2f}', f'{exp2_acc:.2f}']
    
    compare_res = dict(sorted(compare_res.items(), key=lambda x: x[1][0]))
    print(compare_res[:5])
    print(compare_res[-5:-1])

if __name__ == '__main__':
    main()

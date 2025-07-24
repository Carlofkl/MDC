

## Installation
Create a conda environment with the following command:
```bash
conda env create -f environment.yml
```
If this takes too long, `conda config --set solver libmamba` sets conda to use the `libmamba` solver and could speed up installation.

## Running

```bash
python eval_prob_adaptive.py --dataset cifar10 --split test --n_trials 1 \
  --to_keep 5 1 --n_samples 50 500 --loss l1 \
  --prompt_path prompts/cifar10_prompts.csv
```
This command reads potential prompts from a csv file and evaluates the epsilon prediction loss for each prompt using Stable Diffusion.
This should work on a variety of GPUs, from as small as a 2080Ti or 3080 to as large as a 3090 or A6000. 
Losses are saved separately for each test image in the log directory. For the command above, the log directory is `data/cifar10/v2-1_1trials_10_5_1keep_50_100_500samples_l1`. Accuracy can be computed by running:
```bash
python scripts/exp_acc.py --folder data/cifar10/v2-1_1trials_10_5_1keep_50_100_500samples_l1
```

Commands to run Diffusion Classifier on each dataset are [here](commands.md). 
If evaluation on your use case is taking too long, there are a few options: 
1. Parallelize evaluation across multiple workers. Try using the `--n_workers` and `--worker_idx` flags.
2. Play around with the evaluation strategy (e.g. `--n_samples` and `--to_keep`).
3. Evaluate on a smaller subset of the dataset. Saving a npy array of test set indices and using the `--subset_path` flag can be useful for this.


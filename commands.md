### CIFAR-10
```bash
python eval_prob_adaptive.py --dataset cifar10 --split test --n_trials 1 \
  --to_keep 5 1 --n_samples 50 500 --loss l1 \
  --prompt_path prompts/cifar10_prompts.csv
```

### STL-10
```bash
python eval_prob_adaptive.py --dataset stl10 --split test --n_trials 1 \
  --to_keep 5 1 --n_samples 100 500 --loss l1 \
  --prompt_path prompts/stl10_prompts.csv
```

### ImageNet
```bash
python eval_prob_adaptive.py --dataset imagenet --split test --n_trials 1 \
  --to_keep 500 50 10 1 --n_samples 50 100 500 1000 \
  --prompt_path prompts/imagenet_prompts.csv
```

Note: for computational reasons, we evaluated on 4 images per class (4000 test images total).

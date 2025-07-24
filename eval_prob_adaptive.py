import argparse
import numpy as np
import os
import os.path as osp
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from diffusion.datasets import get_target_dataset
from diffusion.models import get_sd_model, get_scheduler_config
from diffusion.utils import LOG_DIR, get_formatstr
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode

import types
from diffusers.models.attention_processor import Attention

device = "cuda" if torch.cuda.is_available() else "cpu"
atn_res = 16

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}




class MyAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atten = None
        self.flag = 'other'

    def get_attention_map(self, hidden_states, encoder_hidden_states):
        query = self.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.norm_cross:
            encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)
        
        key = self.to_k(encoder_hidden_states)

        if query.shape[1] != atn_res * atn_res:
            return None

        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        
        atten = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        del baddbmm_input

        return atten.detach()

    def forward(self, hidden_states, encoder_hidden_states, attention_mask, **cross_attention_kwargs):
        
        self.atten = self.get_attention_map(hidden_states, encoder_hidden_states)

        return super().forward(hidden_states, encoder_hidden_states, attention_mask, **cross_attention_kwargs)

def init_attention_func(unet):

    for name, module in unet.named_modules():
        module_name = type(module).__name__

        if module_name == "Attention":
            parent = unet
            sub_names = name.split('.')
            for sub_name in sub_names[:-1]:
                parent = getattr(parent, sub_name)
            
            original = getattr(parent, sub_names[-1])
            
            my_attn = MyAttention(
                query_dim=original.query_dim,
                cross_attention_dim=original.cross_attention_dim,
                heads=original.heads,
                dropout=original.dropout,
                upcast_attention=original.upcast_attention,
            )
            my_attn = my_attn.to(device=device, dtype=original.to_q.weight.dtype)

            setattr(parent, sub_names[-1], my_attn)
            my_attn.flag = 'self' if name.endswith('attn1') else 'cross'

    return unet

def extract_attention(unet, noise, noise_pred, args):
    gamma = args.gamma
    alpha = args.alpha
    beta = args.beta
    self_atn = []
    cros_atn = []
    for _, module in unet.named_modules():
        if type(module).__name__ == 'MyAttention':
            if module.flag == 'self' and module.atten is not None:
                self_atn.append(module.atten)
            elif module.flag == 'cross' and module.atten is not None:
                cros_atn.append(module.atten)
            else:
                continue
    
    if args.mode == 'cross':
        attention_weight = torch.stack(cros_atn, dim=0).mean(dim=0)

    elif args.mode == 'cross+self':
        tmp = [torch.bmm(a, b) for a, b in zip(self_atn, cros_atn)]
        attention_weight = torch.stack(tmp, dim=0).mean(dim=0)

    tmp = []
    for ids, atten_idx in enumerate(args.atten_idx_input):
        tmp.append(torch.mean(attention_weight[ids:ids+1, :, atten_idx[0]: atten_idx[1]+1], dim=2))  # [1, 1024, 1]
        
    attention_weight = torch.softmax(torch.stack(tmp, dim=0), dim=1).view(-1, atn_res, atn_res)
    attention_weight = F.interpolate(attention_weight.unsqueeze(1), size=(noise.shape[2], noise.shape[3]), mode='bilinear', align_corners=False)
    

    noise = noise * attention_weight * gamma
    noise_pred = noise_pred * attention_weight * gamma
    return noise, noise_pred


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform


def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)


def eval_prob_adaptive(unet, latent, text_embeds, scheduler, args, latent_size=64, all_noise=None):
    scheduler_config = get_scheduler_config(args)
    T = scheduler_config['num_train_timesteps']
    max_n_samples = max(args.n_samples)

    if all_noise is None:
        all_noise = torch.randn((max_n_samples * args.n_trials, 4, latent_size, latent_size), device=latent.device)
    if args.dtype == 'float16':
        all_noise = all_noise.half()
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()

    data = dict()
    t_evaluated = set()
    remaining_prmpt_idxs = list(range(len(text_embeds)))
    start = T // max_n_samples // 2
    t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples]

    for n_samples, n_to_keep in zip(args.n_samples, args.to_keep):
        ts = []
        noise_idxs = []
        text_embed_idxs = []
        curr_t_to_eval = t_to_eval[len(t_to_eval) // n_samples // 2::len(t_to_eval) // n_samples][:n_samples]
        curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]
        for prompt_i in remaining_prmpt_idxs:
            for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):
                ts.extend([t] * args.n_trials)
                noise_idxs.extend(list(range(args.n_trials * t_idx, args.n_trials * (t_idx + 1))))
                text_embed_idxs.extend([prompt_i] * args.n_trials)
        t_evaluated.update(curr_t_to_eval)
        pred_errors = eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                 text_embeds, text_embed_idxs, args.batch_size, args.dtype, args.loss, args)
        # match up computed errors to the data
        for prompt_i in remaining_prmpt_idxs:
            mask = torch.tensor(text_embed_idxs) == prompt_i
            prompt_ts = torch.tensor(ts)[mask]
            prompt_pred_errors = pred_errors[mask]
            if prompt_i not in data:
                data[prompt_i] = dict(t=prompt_ts, pred_errors=prompt_pred_errors)
            else:
                data[prompt_i]['t'] = torch.cat([data[prompt_i]['t'], prompt_ts])
                data[prompt_i]['pred_errors'] = torch.cat([data[prompt_i]['pred_errors'], prompt_pred_errors])

        # compute the next remaining idxs
        errors = [-data[prompt_i]['pred_errors'].mean() for prompt_i in remaining_prmpt_idxs]
        best_idxs = torch.topk(torch.tensor(errors), k=n_to_keep, dim=0).indices.tolist()
        remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs]

    # organize the output
    assert len(remaining_prmpt_idxs) == 1
    pred_idx = remaining_prmpt_idxs[0]

    return pred_idx, data


def eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
               text_embeds, text_embed_idxs, batch_size=32, dtype='float32', loss='l2', args=None):
    assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
    pred_errors = torch.zeros(len(ts), device='cpu')
    idx = 0
    with torch.inference_mode():
        for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
            batch_ts = torch.tensor(ts[idx: idx + batch_size])
            noise = all_noise[noise_idxs[idx: idx + batch_size]]
            noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                            noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
            t_input = batch_ts.to(device).half() if dtype == 'float16' else batch_ts.to(device)
            text_input = text_embeds[text_embed_idxs[idx: idx + batch_size]]
            if args.mode in ['cross', 'cross+self']:
                args.atten_idx_input = [args.attens_idx[l] for l in text_embed_idxs[idx: idx + batch_size]]
            noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
            if args.mode in ['cross', 'cross+self']:
                noise, noise_pred = extract_attention(unet, noise, noise_pred, args)
            if loss == 'l2':
                error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            elif loss == 'l1':
                error = F.l1_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            elif loss == 'huber':
                error = F.huber_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            else:
                raise NotImplementedError
            pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
            idx += len(batch_ts)
    return pred_errors


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', type=str, default='pets',
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')

    # run args
    parser.add_argument('--version', type=str, default='1-5', help='Stable Diffusion model version')
    parser.add_argument('--model_path', type=str, default='../MDC/models/stable-diffusion-v1-5', help='Stable Diffusion model path')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Number of trials per timestep')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--prompt_path', type=str, required=True, help='Path to csv file with prompts to use')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise to use')
    parser.add_argument('--subset_path', type=str, default=None, help='Path to subset of images to evaluate')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    parser.add_argument('--loss', type=str, default='l2', choices=('l1', 'l2', 'huber'), help='Type of loss to use')

    # args for adaptively choosing which classes to continue trying
    parser.add_argument('--to_keep', nargs='+', type=int, required=True)
    parser.add_argument('--n_samples', nargs='+', type=int, required=True)

    parser.add_argument('--gamma', type=float, default=0.001, help='Gamma value for the scheduler')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value for the scheduler')
    parser.add_argument('--beta', type=float, default=0.1, help='Beta value for the scheduler')
    parser.add_argument('--mode', type=str, default='cross', choices=['cross', 'cross+self'],)

    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)

    # make run output folder
    name = f"v{args.version}_{args.n_trials}trials_"
    name += '_'.join(map(str, args.to_keep)) + 'keep_'
    name += '_'.join(map(str, args.n_samples)) + 'samples'
    if args.interpolation != 'bicubic':
        name += f'_{args.interpolation}'
    if args.loss == 'l1':
        name += '_l1'
    elif args.loss == 'huber':
        name += '_huber'
    if args.img_size != 512:
        name += f'_{args.img_size}'
    if args.extra is not None:
        run_folder = osp.join(LOG_DIR, args.dataset + '_' + args.extra, name)
    else:
        run_folder = osp.join(LOG_DIR, args.dataset, name)
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')

    # set up dataset and prompts
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)
    latent_size = args.img_size // 8
    target_dataset = get_target_dataset(args.dataset, train=args.split == 'train', transform=transform)
    prompts_df = pd.read_csv(args.prompt_path)

    # load pretrained models
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    unet = init_attention_func(unet).to(device)
    torch.backends.cudnn.benchmark = True

    # load noise
    if args.noise_path is not None:
        assert not args.zero_noise
        all_noise = torch.load(args.noise_path).to(device)
        print('Loaded noise from', args.noise_path)
    else:
        all_noise = None

    # refer to https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L276
    text_input = tokenizer(prompts_df.prompt.tolist(), padding="max_length",
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    sentences = prompts_df.prompt.tolist()
    attens_idx = []
    for sent_id, sent in enumerate(sentences):
        if args.dataset in ['food']:
            sent = sent[sent.index('of')+3: min(sent.index('.'), sent.index(','))].strip()
        elif args.dataset in ['cifar10', 'stl10', 'cifar100', 'toy', 'caltech101']:
            sent = sent[sent.index('of')+5: sent.index('.')].strip()
        elif args.dataset in ['mnist']:
            sent = sent[sent.index(':')+3: sent.index('.')-1].strip()
        tokens = tokenizer([sent])['input_ids'][0][1:-1]
        prompt_token = text_input['input_ids'][sent_id].tolist()
        start_idx, end_idx = prompt_token.index(tokens[0]), prompt_token.index(tokens[-1])
        attens_idx.append([start_idx, end_idx])
    args.attens_idx = attens_idx
    embeddings = []
    with torch.inference_mode():
        for i in range(0, len(text_input.input_ids), 100):
            text_embeddings = text_encoder(
                text_input.input_ids[i: i + 100].to(device),
            )[0]
            embeddings.append(text_embeddings)
    text_embeddings = torch.cat(embeddings, dim=0)
    assert len(text_embeddings) == len(prompts_df) == len(attens_idx)

    # subset of dataset to evaluate
    if args.subset_path is not None:
        idxs = np.load(args.subset_path).tolist()
    else:
        idxs = list(range(len(target_dataset)))
    idxs_to_eval = idxs[args.worker_idx::args.n_workers]

    formatstr = get_formatstr(len(target_dataset) - 1)
    correct = 0
    total = 0
    pbar = tqdm.tqdm(idxs_to_eval)
    for i in pbar:
        if total > 0:
            pbar.set_description(f'Acc: {100 * correct / total:.2f}%')
        fname = osp.join(run_folder, formatstr.format(i) + '.pt')
        if os.path.exists(fname):
            print('Skipping', i)
            if args.load_stats:
                data = torch.load(fname)
                correct += int(data['pred'] == data['label'])
                total += 1
            continue
        image, label = target_dataset[i]
        with torch.no_grad():
            img_input = image.to(device).unsqueeze(0)
            if args.dtype == 'float16':
                img_input = img_input.half()
            x0 = vae.encode(img_input).latent_dist.mean
            x0 *= 0.18215
        pred_idx, pred_errors = eval_prob_adaptive(unet, x0, text_embeddings, scheduler, args, latent_size, all_noise)
        pred = prompts_df.classidx[pred_idx]
        torch.save(dict(errors=pred_errors, pred=pred, label=label), fname)
        if pred == label:
            correct += 1
        total += 1


if __name__ == '__main__':
    main()

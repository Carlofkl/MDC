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
import matplotlib.pyplot as plt
import cv2

import random
seed = 0
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

import time

# device = "cuda:0" if torch.cuda.is_available() else "cpu"

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

def save_fig(data, fname, type):
    os.makedirs(fname, exist_ok=True)
    torch.save(data, osp.join(fname, type + '.pt'))
    # plt.clf()
    # plt.axis('off')
    
    # if type == 'img':
    #     plt.imshow(data)
    # elif type == 'img_bi':
    #     plt.imshow(data, cmap='cividis')
    # else:
    #     plt.imshow(data, cmap='cividis')
        
    # file_to_save = osp.join(fname, type + '.png')
    # plt.savefig(file_to_save, bbox_inches='tight', pad_inches=0.001)
    # plt.close()

def save_atten(img, img_atn, attentions, vis_path):
    # input [bs, 1, 64, 64]
    # print('you have been here')
    
    # save image
    save_fig(img, vis_path, 'img')
    
    # save img-bi
    img_atn = img_atn.reshape(1, 64, 64)
    img_gau = cv2.GaussianBlur(img_atn.cpu().numpy(), (3, 3), 0.5)
    thres = np.mean(img_gau)
    img_bi = (img_gau < thres).astype(int)
    save_fig(img_bi[0], vis_path, 'img_bi')
    
    for cls_k, cls_v in attentions.items():
        vis_folder = osp.join(vis_path, f'prompt_{cls_k}')
        os.makedirs(vis_folder, exist_ok=True)
        for k, v in cls_v.items():
            if int(k) > 1000:
                break
            attention_to_save = [v[0].cpu(), v[1].cpu(), v[2].cpu(), v[3].cpu()]

            # fig, axes = plt.subplots(1, 5, figsize=(10, 2), constrained_layout=True)
            # for idx, ax in enumerate(axes.flat):
            #     ax.imshow(attention_to_save[idx])
            #     ax.axis('off')
            for i in range(len(attention_to_save)):
                atn_gau = cv2.GaussianBlur(attention_to_save[i].numpy(), (3, 3), 0.5)
                save_fig(atn_gau[0], vis_folder, f"{k}_{i}")
            #     plt.imshow(attention_to_save[i])
            #     file_to_save = os.path.join(vis_folder, f"{i}.png")
            #     plt.savefig(file_to_save, bbox_inches='tight', pad_inches=0.001)
            # plt.close()

def clean(attentions, atten_idx_input, add_atten, alpha=1.):
    
    downattentions = attentions[0]
    upattentions=attentions[2]

    # mean & scale 
    cros_downattentions = torch.stack([x for x in downattentions['cros'] if x is not None], dim=0)  # [2, bs, 1024, 77]
    cros_downattentions = torch.mean(cros_downattentions/10, dim=0)  # [bs, 1024, 77]
    self_downattentions = torch.stack([x for x in downattentions['self'] if x is not None], dim=0)  # [2, bs, 1024, 1024]
    self_downattentions = torch.mean(self_downattentions/10, dim=0)  # [bs, 1024, 1024]
    
    cros_upattentions = torch.stack([x for x in upattentions['cros'] if x is not None], dim=0)  # [3, bs, 1024, 77]
    cros_upattentions = torch.mean(cros_upattentions/10, dim=0)  # [bs, 1024, 77]
    self_upattentions = torch.stack([x for x in upattentions['self'] if x is not None], dim=0)  # [3, bs, 1024, 1024]
    self_upattentions = torch.mean(self_upattentions/10, dim=0)  # [bs, 1024, 1024]
    
    # get targeted token attention
    cross_downattens, cross_upattens = [], []
    for ids, atten_idx in enumerate(atten_idx_input):
        cross_downattens.append(torch.mean(cros_downattentions[ids:ids+1, :, atten_idx[0]: atten_idx[1]+1], dim=2, keepdim=True))  # [1, 1024, 1]
        cross_upattens.append(torch.mean(cros_upattentions[ids:ids+1, :, atten_idx[0]: atten_idx[1]+1], dim=2, keepdim=True))
    cross_downattens = torch.cat(cross_downattens, dim=0)  # [bs, 1024, 1]
    cross_upattens = torch.cat(cross_upattens, dim=0)  # [bs, 1024, 1]
    
    # self attention is too large
    # self_downattentions, self_upattentions = torch.ones(1,1), torch.ones(1,1) 
    return [cross_downattens, cross_upattens, self_downattentions, self_upattentions]
    
def fusion(attentions, mask, add_atten, alpha, beta):
    # [bs, 1024, 1], [bs, 1024, 1], [bs, 1024, 1024], [bs, 1024, 1024]
    
    cros_downattention, cros_upattention, self_downattention, self_upattention = attentions[0], attentions[1], attentions[2], attentions[3]
    
    # cros_attention, self_attention = attentions[0], attentions[1]
    # cros_downattention, cros_upattention = torch.ones_like(cros_attention), torch.ones_like(cros_attention)
    # cros_attention = cros_attention.reshape(cros_attention.shape[0], 1, 32, 32)  # [2, 1, 32, 32]
    # cros_attention = F.interpolate(cros_attention, size=(64, 64), mode='bilinear')  # [2, 1, 64, 64]
    # cros_attention = cros_attention.reshape(cros_attention.shape[0], 4096, 1)
    
    # fuse down & up
    cros_attention = alpha * cros_downattention + (1-alpha) * cros_upattention  # [bs, 1024, 1]
    a0max, a0min, a0mean, a0sum = torch.max(cros_attention), torch.min(cros_attention), torch.mean(cros_attention), torch.sum(cros_attention)

    # fuse cross & self
    # softmax
    if add_atten == 4:
        self_attention = alpha * self_downattention + (1-alpha) * self_upattention  # [bs, 1024, 1024]
        # cros [bs, 1024, 1], self [bs, 1024, 1024]
        scatten = (0.25 * (torch.ones_like(self_attention) + self_attention + self_attention**2 + self_attention**3) @ cros_attention)  # [bs, 4096, 1]
        attention = F.softmax(attention/100, dim=-1)  # /1000 or /100 ?????
        attention = (scatten - torch.min(scatten)) / (torch.max(scatten) - torch.min(scatten))
        # TODO
    else:
        attention = F.softmax(cros_attention, dim=1)
        # attention = (cros_attention - torch.min(cros_attention)) / (torch.max(cros_attention) - torch.min(cros_attention))
        a1max, a1min, a1mean, a1sum = torch.max(attention), torch.min(attention), torch.mean(attention), torch.sum(attention)
        
    # vis part
    d_attention = F.softmax(cros_downattention, dim=1)
    u_attention = F.softmax(cros_upattention, dim=1)
    du_attention = F.softmax(cros_attention, dim=1)
    self_attention = alpha * self_downattention + (1-alpha) * self_upattention
    self_attention = (self_attention - torch.min(self_attention)) / (torch.max(self_attention) - torch.min(self_attention))
    
    # reshape 
    hw = int(attention.shape[1] ** 0.5)
    atten = attention.reshape(attention.shape[0], 1, hw, hw)  # [2, 1, 32, 32]
    d_attention = d_attention.reshape(d_attention.shape[0], 1, hw, hw)
    u_attention = u_attention.reshape(u_attention.shape[0], 1, hw, hw)
    du_attention = du_attention.reshape(du_attention.shape[0], 1, hw, hw)
    self_attention = self_attention.unsqueeze(1)
    
    # resize [2, 1, 64, 64]
    # atten_rsz = atten
    # d_attention, u_attention, du_attention = torch.ones_like(atten_rsz), torch.ones_like(atten_rsz), torch.ones_like(atten_rsz)
    atten_rsz = F.interpolate(atten, size=(64, 64), mode='bilinear')
    d_attention = F.interpolate(d_attention, size=(256, 256), mode='bilinear')
    u_attention = F.interpolate(u_attention, size=(256, 256), mode='bilinear')
    du_attention = F.interpolate(du_attention, size=(256, 256), mode='bilinear')
    self_attention = F.interpolate(self_attention, size=(256, 256), mode='bilinear')
    
    # fusion with ones
    if add_atten in [3, 5]:
        sc_du_atten = mask + beta * atten_rsz
    elif add_atten == 4:
        sc_du_atten = beta * atten_rsz
    
    attens_to_save = [d_attention, u_attention, du_attention, self_attention]
    return sc_du_atten, attens_to_save

def read_mid_results(mid_res, idx, device):
    # {'noise': noise, 'pred': pred, 'atns': atn}
    noise, pred, attentions = mid_res[idx]['noise'], mid_res[idx]['pred'], mid_res[idx]['atns']
    noise_gt = noise.to(device)
    noise_pred = pred.to(device)
    attentions_ = [x.to(device) for x in attentions]
    return noise_gt, noise_pred, attentions_

def cal_atn(img_atn, maps, batch_t):
    atn_dis = torch.zeros_like(batch_t).to(batch_t.device).float()
    for bs_idx, t in enumerate(batch_t.numpy()):
        map = maps[bs_idx].reshape(1, 64, 64)
        if t > 100:
            image_atn = img_atn.reshape(1, 64, 64)
            map_gau = cv2.GaussianBlur(map.cpu().numpy(), (3, 3), 0.5)
            img_gau = cv2.GaussianBlur(image_atn.cpu().numpy(), (3, 3), 0.5)
            
            map_bi = (map_gau < np.median(map_gau)).astype(int)
            img_bi = (img_gau < np.median(img_gau)).astype(int)
        
            # fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
            # axes.flat[0].imshow(map_bi[0])
            # axes.flat[1].imshow(img_bi[0])
            # plt.savefig(f'atn_bi_{t}.png')
            # plt.close()
        
            atn_dis[bs_idx] = np.mean((img_bi ^ map_bi))
        
    return atn_dis

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


def eval_prob_adaptive(unet, latent, text_embeds, attens_idx, scheduler, args, latent_size=64, all_noise=None, offline_debug={'flag': False, 'path': '_'}, img_atn=None):
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

    if offline_debug['flag'] and img_atn is not None:
        os.makedirs(offline_debug['path'], exist_ok=True)
        torch.save(img_atn, osp.join(offline_debug['path'], 'img_atn.pt'))

    attentions_to_save = {}
    for n_samples, n_to_keep in zip(args.n_samples, args.to_keep):
        offlinedebug = {'flag': offline_debug['flag'],'path': osp.join(offline_debug['path'], f'{n_to_keep}_{n_samples}.pt')}
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
        # save mid-results
        mid_res= []
        if offlinedebug['flag'] and os.path.exists(offlinedebug['path']):
            mid_res = torch.load(offlinedebug['path'])
            
        pred_errors, attentions, keep_trail = eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                            text_embeds, text_embed_idxs, attens_idx, args.batch_size, args.dtype, args.loss, args.add_atten, args.alpha, args.beta, mid_res, offlinedebug['flag'], img_atn, args.gamma)
        if offlinedebug['flag'] and len(mid_res) == 0:
            torch.save(keep_trail, offlinedebug['path'])
        
        if len(attentions_to_save.keys()) == 0:
            attentions_to_save.update(attentions)
            
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

    return pred_idx, data, attentions_to_save


def eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
               text_embeds, text_embed_idxs, attens_idx, batch_size=32, dtype='float32', loss='l2', add_atten=0, alpha=1., beta=0.5, mid_res=[], save_mid_res=False, img_atn=None, gamma=0.):
    assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
    pred_errors = torch.zeros(len(ts), device='cpu')
    idx = 0
    idx_set = set(text_embed_idxs)
    attentions_to_save = {str(k): {} for k in idx_set}
    keep_trail = []
    with torch.inference_mode():
        for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
            batch_ts = torch.tensor(ts[idx: idx + batch_size])
            noise = all_noise[noise_idxs[idx: idx + batch_size]]
            noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(unet.device) + \
                            noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(unet.device)
            t_input = batch_ts.to(unet.device).half() if dtype == 'float16' else batch_ts.to(unet.device)
            text_input = text_embeds[text_embed_idxs[idx: idx + batch_size]]
            atten_idx_input = [attens_idx[l] for l in text_embed_idxs[idx: idx + batch_size]]
            
            if len(mid_res) > 0:
                noise_gt, noise_pred, attentions = read_mid_results(mid_res, _, t_input.device)
                noise = noise_gt
            else:
                unet_output, attentions = unet(noised_latent, t_input, encoder_hidden_states=text_input)
                attentions = clean(attentions, atten_idx_input, add_atten, alpha)
                noise_pred = unet_output.sample  # [2, 4, 64, 64]
                if save_mid_res:
                    keep_trail.append({'noise': noise.cpu(), 'pred': noise_pred.cpu(), 'atns': [x.cpu() for x in attentions]})
                
            # data cleaning
            maps = torch.ones(noise_pred.shape[0], 1, noise_pred.shape[2], noise_pred.shape[3], device=unet.device)
            if add_atten != 0:
                maps_new, attens_to_save = fusion(attentions, maps, add_atten, alpha, beta)  # downatten [2, 1, 64, 64]
                for bs_idx, t in enumerate(batch_ts.numpy()):
                    if t % 15 == 0:
                        cls_idx = text_embed_idxs[idx: idx + batch_size][bs_idx]
                        attentions_to_save[str(cls_idx)].update(
                            {str(t): [attens_to_save[0][bs_idx], attens_to_save[1][bs_idx], attens_to_save[2][bs_idx], attens_to_save[3][bs_idx]]}
                        ) 
            if add_atten >= 3:
                maps = maps_new
            elif add_atten == 4:
                noise_pred += maps_new
            
            atn_loss = torch.zeros_like(batch_ts)
            if add_atten == 5:
                atn_loss = cal_atn(img_atn, maps_new, batch_ts)
            atn_loss = atn_loss.to(unet.device)
            
            if loss == 'l2':
                error = (F.mse_loss(noise, noise_pred, reduction='none') * maps).mean(dim=(1, 2, 3)) + gamma * atn_loss
            elif loss == 'l1':
                error = (F.l1_loss(noise, noise_pred, reduction='none') * maps).mean(dim=(1, 2, 3)) + gamma * atn_loss
            elif loss == 'huber':
                error = (F.huber_loss(noise, noise_pred, reduction='none') * maps).mean(dim=(1, 2, 3)) + gamma * atn_loss
            else:
                raise NotImplementedError
            pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
            idx += len(batch_ts)        
    return pred_errors, attentions_to_save, keep_trail


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', type=str, default='toy',
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft', 'cifar100', 'toy'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Name of split')

    # run args
    parser.add_argument('--version', type=str, default='2-1', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Number of trials per timestep')
    parser.add_argument('--batch_size', '-b', type=int, default=2)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--prompt_path', type=str, default='prompts/stl10_prompts.csv', help='Path to csv file with prompts to use')
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
    parser.add_argument('--to_keep', nargs='+', type=int)
    parser.add_argument('--n_samples', nargs='+', type=int)
    
    parser.add_argument('--add_atten', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1.0)  # alpha=1, only use downatten
    parser.add_argument('--beta', type=float, default=0.5)  # smaller, weaken attention's influence
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--vis', action='store_true')  # to debug and visualize
    parser.add_argument('--device', type=int, default=0, help='choose which gpu')
    parser.add_argument('--offline_debug', action='store_true')

    args = parser.parse_args()
    
    # stick to debug
    args.to_keep = [5, 1]
    args.n_samples = [100, 500]
    args.loss = 'l1'
    # args.vis = True
    args.add_atten = 3
    args.alpha = 0.5
    args.beta = -100.0
    args.device = 1
    # args.offline_debug = True
    
    device = f"cuda:{args.device}"
    assert len(args.to_keep) == len(args.n_samples)

    # make run output folder
    name = f"v{args.version}_{args.n_trials}trials_"
    name += '_'.join(map(str, args.to_keep)) + 'keep_'
    name += '_'.join(map(str, args.n_samples)) + 'samples'
    mid_res_name = name + f'_{args.batch_size}'
    if args.interpolation != 'bicubic':
        name += f'_{args.interpolation}'
    if args.loss == 'l1':
        name += '_l1'
    elif args.loss == 'huber':
        name += '_huber'
    if args.img_size != 512:
        name += f'_{args.img_size}'
    if args.add_atten == 1:  # += (up * down)
        name += '_DUatten'
    elif args.add_atten == 2:  # alpha = 1 调试中
        name += '_aDpUatten' + f"_alpha={args.alpha}_beta={args.beta}"
    elif args.add_atten == 3:
        name += '_mDpUatten' + f"_alpha={args.alpha}_beta={args.beta}"
    elif args.add_atten == 4:
        name += '_SCDpUatten' + f"_alpha={args.alpha}_beta={args.beta}"
    elif args.add_atten == 5:
        name += '_AtnDis' + f"_alpha={args.alpha}_beta={args.beta}_gamma={args.gamma}"
        
    if args.vis:
        name_vis = name + '_vis'
        
    if args.extra is not None:
        run_folder = osp.join(LOG_DIR, args.dataset + '_' + args.extra, name)
    else:
        run_folder = osp.join(LOG_DIR, args.dataset, name)
        if args.vis:
            # run_folder_vis = osp.join(LOG_DIR, args.dataset, name_vis)
            run_folder_vis = osp.join("Figures", args.dataset)

    mid_res_folder = osp.join('mid_results', args.dataset, mid_res_name)
    
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')
    if args.vis:
        print(f'Vis folder: {run_folder_vis}')
    if args.offline_debug:
        print(f'Read mid folder: {mid_res_folder}')

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
    sents = prompts_df.prompt.tolist()
    attens_idx = []
    for sent_id, sent in enumerate(sents):
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
        # idxs = list(range(len(target_dataset)))
        idxs = list(range(target_dataset.__len__()))
    idxs_to_eval = idxs[args.worker_idx::args.n_workers]

    # formatstr = get_formatstr(len(target_dataset) - 1)
    formatstr = get_formatstr(target_dataset.__len__() - 1)
    correct = 0
    total = 0
    pbar = tqdm.tqdm(idxs_to_eval)
    for i in pbar:
        if total > 0:
            pbar.set_description(f'Acc: {100 * correct / total:.2f}%')
        fname = osp.join(run_folder, formatstr.format(i) + '.pt')
        offlinedebug = {'flag': args.offline_debug, 'path': osp.join(mid_res_folder, formatstr.format(i))}
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
            x0, img_atn = vae.encode(img_input)
            x0 = 0.18215 * x0.latent_dist.mean
            ##### kailai code #####
            img_atn = torch.mean(img_atn, dim=2)
            img_atn = (img_atn - torch.min(img_atn)) / (torch.max(img_atn) - torch.min(img_atn))
            # print(img_atn)
            # plt.imshow(img_atn.reshape(64, 64).cpu())
            # plt.savefig(f'test_demo-{i}.png')
            
        pred_idx, pred_errors, attentions = eval_prob_adaptive(unet, x0, text_embeddings, attens_idx, scheduler, args, latent_size, all_noise, offlinedebug, img_atn)
        ##### kailai code #####
        if args.vis:
            imm = np.transpose((image.numpy() + 1) / 2, (1, 2, 0))
            vis_path = osp.join(run_folder_vis, formatstr.format(i))
            save_atten(imm, img_atn, attentions, vis_path)
        pred = prompts_df.classidx[pred_idx]
        torch.save(dict(errors=pred_errors, pred=pred, label=label), fname)
        if pred == label:
            correct += 1
        total += 1


if __name__ == '__main__':
    # print('start')
    # time.sleep(18000)
    main()

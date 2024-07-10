import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from diffusion.models import get_sd_model, get_scheduler_config
import argparse
from eval_prob_adaptive import get_formatstr, get_transform, get_target_dataset, INTERPOLATIONS, eval_prob_adaptive, save_atten
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import os.path as osp
import cv2
# Generate Gaussian noise
# size = 128
# noise = np.random.normal(0, 1, (size, size))


# # Display the image
# plt.axis('off')
# plt.imshow(noise, cmap='cividis')

# # Save the image
# plt.savefig(f'Figures/gaussian_noise_{size}.png', bbox_inches='tight', pad_inches=0.001)

# exit()





def _convert_image_to_rgb(image):
    return image.convert("RGB")
# Load STL10 dataset
# transform = torchvision.transforms.Compose([
#     # torchvision.transforms.ToTensor(),
#     # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     torch_transforms.Resize(512, interpolation=InterpolationMode.BICUBIC),
#     torch_transforms.CenterCrop(512),
#     _convert_image_to_rgb,
#     torch_transforms.ToTensor(),
#     torch_transforms.Normalize([0.5], [0.5])
# ])
# trainset = torchvision.datasets.STL10(root='datasets', split='test', download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)

# Define function to show image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('Figures/stl10_1.png')
    plt.show()


def save_fig(data, fname, type):
    plt.clf()
    plt.axis('off')
    
    if type == 'img':
        plt.imshow(data)
    elif type == 'img_atn':
        image_atn = data.reshape(1, 64, 64)
        img_gau = cv2.GaussianBlur(image_atn.cpu().numpy(), (3, 3), 0.5)
        thres = np.mean(img_gau ** 2)
        img_bi = (img_gau < thres).astype(int)
        plt.imshow(img_bi[0], cmap='cividis')
    
    file_to_save = osp.join(fname, type + '.png')
    plt.savefig(file_to_save, bbox_inches='tight', pad_inches=0.001)
    plt.close()



device = 'cuda:1'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='stl10')
parser.add_argument('--version', type=str, default='2-1')    
parser.add_argument('--device', type=str, default='cuda:1')           
parser.add_argument('--prompt_path', type=str, default='prompts/stl10_prompts.csv')                  
parser.add_argument('--dtype', type=str, default='float16')                  
parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
parser.add_argument('--img_size', type=int, default=512)         

args = parser.parse_args()
args.dype = 'float16'
args.version = '2-1'
args.device = device
args.prompt_path = 'prompts/stl10_prompts.csv'
args.dataset = 'stl10'
run_folder = 'Figures/stl10'

# prepare model
vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
vae = vae.to(device)
text_encoder = text_encoder.to(device)
unet = unet.to(device)
torch.backends.cudnn.benchmark = True

# prepare data
interpolation = INTERPOLATIONS[args.interpolation]
transform = get_transform(interpolation, args.img_size)
latent_size = args.img_size // 8
target_dataset = get_target_dataset(args.dataset, train=False, transform=transform)
prompts_df = pd.read_csv(args.prompt_path)

text_input = tokenizer(prompts_df.prompt.tolist(), padding="max_length",
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
sents = prompts_df.prompt.tolist()
attens_idx = []
for sent_id, sent in enumerate(sents):
    if args.dataset in ['food']:
        sent = sent[sent.index('of')+3: min(sent.index('.'), sent.index(','))].strip()
    elif args.dataset in ['cifar10', 'stl10', 'cifar100', 'toy', 'caltech101']:
        sent = sent[sent.index('of')+5: sent.index('.')].strip()

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

# run
formatstr = get_formatstr(target_dataset.__len__() - 1)
idxs_to_eval = list(range(target_dataset.__len__()))
pbar = tqdm.tqdm(idxs_to_eval)

for i in pbar:
    fname = osp.join(run_folder, formatstr.format(i))
    os.makedirs(fname, exist_ok=True)
    
    if os.path.exists(osp.join(fname, 'atn.png')):
        print('Skipping', i)

    image, label = target_dataset[i]
    img = np.transpose((image.numpy() + 1) / 2, (1, 2, 0))
    save_fig(img, fname, 'img')
    
    offlinedebug = {'flag': True, 'path': osp.join('mid_results/toy/v2-1_1trials_5_1keep_100_500samples_2', formatstr.format(i))}
    
    with torch.no_grad():
        img_input = image.to(device).unsqueeze(0)
        if args.dtype == 'float16':
            img_input = img_input.half()
        x0, img_atn = vae.encode(img_input)
        x0 = 0.18215 * x0.latent_dist.mean

        img_atn = torch.mean(img_atn, dim=2)
        img_atn = (img_atn - torch.min(img_atn)) / (torch.max(img_atn) - torch.min(img_atn))
        save_fig(img_atn, fname, 'img_atn')
    
    # pred_idx, pred_errors, attentions = eval_prob_adaptive(unet, x0, text_embeddings, attens_idx, scheduler, args, latent_size, None, offlinedebug, img_atn)

    scheduler_config = get_scheduler_config(args)
    T = scheduler_config['num_train_timesteps']
    
    data = dict()
    t_evaluated =set()
    remaining_prmpt_idxs = list(range(len(text_embeddings)))
    
    start = T
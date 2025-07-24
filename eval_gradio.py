import torch
import tqdm
import argparse
from PIL import Image
from diffusion.models import get_sd_model, get_scheduler_config
from eval_prob_adaptive import eval_prob_adaptive
from eval_prob_adaptive import get_transform
from torchvision.transforms.functional import InterpolationMode

import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

classes = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
]

def get_prompts_df():
    # prompt, classname, classidx
    prompts_df = []
    for i, c in enumerate(classes):
        prompts_df.append({
            'prompt': f"a photo of a {c}",
            'classname': c,
            'classidx': i
        })

    return prompts_df

def eval_image(target_dataset):

    parser = argparse.ArgumentParser()
     # dataset args
    # run args
    parser.add_argument('--model_path', type=str, default='../MDC/models/stable-diffusion-v1-5', help='Stable Diffusion model path')
    parser.add_argument('--version', type=str, default='1-5', choices=('1-1', '1-2', '1-3', '1-4', '1-5', '2-0', '2-1'))
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Number of trials per timestep')
    parser.add_argument('--batch_size', '-b', type=int, default=8)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    # parser.add_argument('--prompt_path', type=str, required=True, help='Path to csv file with prompts to use')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    parser.add_argument('--loss', type=str, default='l2', choices=('l1', 'l2', 'huber'), help='Type of loss to use')

    # args for adaptively choosing which classes to continue trying
    parser.add_argument('--to_keep', nargs='+', type=int, default=[5, 1])
    parser.add_argument('--n_samples', nargs='+', type=int, default=[50, 500])

    parser.add_argument('--mode', type=str, default='None')

    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)

    # set up data and prompts
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)
    latent_size = args.img_size // 8
    prompts_df = get_prompts_df()

    # load pretrained model
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    torch.backends.cudnn.benchmark = True

    text_input = tokenizer(
        [x['prompt'] for x in prompts_df], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    embeddings = []
    with torch.inference_mode():
        for i in range(0, len(text_input.input_ids), 100):
            text_embeddings = text_encoder(
                text_input.input_ids[i: i + 100].to(device),
            )[0]
            embeddings.append(text_embeddings)
    text_embeddings = torch.cat(embeddings, dim=0)
    assert len(text_embeddings) == len(prompts_df)

    idxs = list(range(len(target_dataset)))
    idxs_to_eval = idxs[args.worker_idx::args.n_workers]

    pbar = tqdm.tqdm(idxs_to_eval)
    results = []
    for i in pbar:
        image, label = target_dataset[i]
        image = transform(image)
        with torch.no_grad():
            img_input = image.to(device).unsqueeze(0)
            if args.dtype == 'float16':
                img_input = img_input.half()
            x0 = vae.encode(img_input).latent_dist.mean
            x0 *= 0.18215
        pred_idx, pred_errors = eval_prob_adaptive(unet, x0, text_embeddings, scheduler, args, latent_size)
        pred = prompts_df[pred_idx]['classname']
        gt = prompts_df[label]['classname']
        if pred_idx == label:
            pbar.set_description(f"Correct: {pred_idx}={pred}, {label}={gt}")
        results.append(pred)

    return results

def run_model(*args):
    # label_text = args[-1]  # 最后一个参数是标签输入
    # labels = label_text.split(",")
    # labels = [label.strip() for label in labels]
    labels = [1] * len(args)

    images = args[:]
    
    assert len(images) == len(labels), f"请上传{len(images)}张图片"

    target_dataset = []
    for img, label_str in zip(images, labels):
        # if img is None:
        #     return ["请上传所有5张图片"] * 5
        label_int = int(label_str)
        target_dataset.append((img, label_int))
    target_dataset = target_dataset
    preds = eval_image(target_dataset)  # 你的函数
    return preds



default_img1 = Image.open("data/demo/dog.jpg")
default_img2 = Image.open("data/demo/plane.png")
default_label = ['dog', 'airplane']

with gr.Blocks() as demo:
    gr.Markdown("## 上传图片，图片类别在airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck之内")

    with gr.Row():
        # img_inputs = [gr.Image(type="pil", label=f"Image {i+1}") for i in range(5)]

        img1 = gr.Image(type="pil", label="Image 1", value=default_img1)
        img2 = gr.Image(type="pil", label="Image 2", value=default_img2)
        # img3 = gr.Image(type="pil", label="Image 3", value=None)
        # img4 = gr.Image(type="pil", label="Image 4", value=None)
        # img5 = gr.Image(type="pil", label="Image 5", value=None)
    
    img_inputs = [img1, img2]
    # label_input = gr.Textbox(
    #     label="标签（逗号分隔，参考如：0,1,2,3,4。对应关系：0-airplane, 1-automobile, 2-bird, 3-cat, 4-deer, 5-dog, 6-frog, 7-horse, 8-ship, 9-truck）", lines=1
    # )

    btn = gr.Button("提交")
    with gr.Row():
        outputs = [gr.Textbox(label=f"预测{i+1}", value=default_label[i]) for i in range(len(img_inputs))]

    btn.click(
        fn=run_model, 
        inputs=img_inputs, 
        outputs=outputs
    )

demo.launch(share=True)

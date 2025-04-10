import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = 'OpenGVLab/InternVL2_5-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# set the max number of tiles in `max_num`
pixel_values = load_image('mm_mapworld/mm_mapworld_main/resources/ade_20k_reduced/ade_imgs/basement/ADE_train_00002481.jpg', max_num=12).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=True)

# pure-text conversation (纯文本对话)
question = 'Hello, who are you?'
response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'Can you tell me a story?'
response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# single-image single-round conversation (单图单轮对话)
question = '<image>\nPlease describe the image shortly.'
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}\nAssistant: {response}')




{'graph_id': '21r22w32p33k23w13c03g02c12b11b', 
 'm': 4, 
 'n': 4, 
 'graph_nodes': ['Restroom outdoor', 'Wine cellar bottle storage', 'Parlor', 'Kitchen', 'Waiting room', 'Computer room', 'Game room', 'Corridor', 'Basement', 'Bistro outdoor'], 
 'graph_edges': [('Restroom outdoor', 'Wine cellar bottle storage'), ('Wine cellar bottle storage', 'Parlor'), ('Parlor', 'Kitchen'), ('Kitchen', 'Waiting room'), ('Waiting room', 'Computer room'), ('Computer room', 'Game room'), ('Game room', 'Corridor'), ('Corridor', 'Basement'), ('Basement', 'Bistro outdoor')], 
 'directions': {'Restroom outdoor': ['north'], 'Wine cellar bottle storage': ['south', 'east'], 'Parlor': ['west', 'north'], 'Kitchen': ['south', 'west'], 'Waiting room': ['east', 'west'], 'Computer room': ['east', 'west'], 'Game room': ['east', 'south'], 'Corridor': ['north', 'east'], 'Basement': ['west', 'south'], 'Bistro outdoor': ['north']}, 
 'moves': {'Restroom outdoor': [('north', 'Wine cellar bottle storage')], 'Wine cellar bottle storage': [('south', 'Restroom outdoor'), ('east', 'Parlor')], 'Parlor': [('west', 'Wine cellar bottle storage'), ('north', 'Kitchen')], 'Kitchen': [('south', 'Parlor'), ('west', 'Waiting room')], 'Waiting room': [('east', 'Kitchen'), ('west', 'Computer room')], 'Computer room': [('east', 'Waiting room'), ('west', 'Game room')], 'Game room': [('east', 'Computer room'), ('south', 'Corridor')], 'Corridor': [('north', 'Game room'), ('east', 'Basement')], 'Basement': [('west', 'Corridor'), ('south', 'Bistro outdoor')], 'Bistro outdoor': [('north', 'Basement')]}, 
 'mapping': {(2, 1): 'Restroom outdoor', (2, 2): 'Wine cellar bottle storage', (3, 2): 'Parlor', (3, 3): 'Kitchen', (2, 3): 'Waiting room', (1, 3): 'Computer room', (0, 3): 'Game room', (0, 2): 'Corridor', (1, 2): 'Basement', (1, 1): 'Bistro outdoor'}
}
import torch, os, json, random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import argparse
from data_aug import data_augmentation
from clip_count_utils import *
from my_datasets import load_dataset

def create_my_own_dataset_texts(
        train_set,
        model,
        processor,
        device="cuda",
        # num_class = 2,
):
    
    my_count_bench_dataset=[]
    for i, sample in tqdm(enumerate(train_set)):
        # if sample["number"] == num_class:
        #     continue
        if sample["image"] is None:
            try:
                image = Image.open(requests.get(sample["image_url"], stream=True,timeout=2).raw)
            except Timeout:
                print("timeout")
            except:
                continue
        else:
            image = sample["image"]
        
        target_obj_aug_with_context_text,target_obj_text,target_obj_with_context_text,target_obj_aug_text = sentence_augmentation(sample["text"])

        pixel_values=processor(text=None, images=image, return_tensors="pt", padding=True)["pixel_values"] # torch.Size([1, 3, 224, 224])
        
        image_embeds = get_image_embeds(
            model=model,
            pixel_values=pixel_values,
            device=device,
        ) 
        my_count_bench_dataset.append(
            {
                "gt_count":sample['number'],
                "target_obj_text":target_obj_text,
                "target_obj_aug_text":target_obj_aug_text,
                "target_obj_with_context_text":target_obj_with_context_text,
                "target_obj_aug_with_context_text":target_obj_aug_with_context_text,
                "image_embeds":image_embeds.detach().cpu(),
            }
        )
        del pixel_values, image_embeds
    return my_count_bench_dataset


def countbench_save_name(model_name,num_class,texts):
    save_name = f"countbench_{model_name}_{num_class}class_texts{texts}.pth"
    print(f"Save name: {save_name}")
    return save_name

device="cuda"
train_set = load_dataset("nielsr/countbench",streaming=True)["train"]
save_root_folder="./countbench/"
texts=True
for model_name in  ["clip-vit-large-patch14"]: #,"clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(f"openai/{model_name}").to(device)
    model.requires_grad=False
    processor = CLIPProcessor.from_pretrained(f"openai/{model_name}")
    # for num_class in [2,3,4,5,6,7,8,9,10]:
        
    my_count_bench = create_my_own_dataset_texts(
            train_set,
            model,
            processor,
            device=device,
            # num_class = num_class,
    )
    torch.save(
        my_count_bench,
        os.path.join(save_root_folder,countbench_save_name(model_name,"all",texts))
    )
    del my_count_bench
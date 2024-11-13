import torch, os, json, random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import argparse
import numpy as np
from data_aug import data_augmentation
from clip_count_utils import *
from sklearn.model_selection import train_test_split

class TextDatasetOnlineTrain(Dataset):
    def __init__(self, data_path, processor, num_classes=4, ratio=1,add_object_cf=False,ref_obj_file=None):
        self.num_classes = num_classes
    
        self.true_texts, self.cf_texts, self.image_embeds, self.target_obj_texts, self.gt_label_idx, self.cf_label_idx = self.create_dataset(data_path,ratio)

        assert len(self.true_texts) == len(self.cf_texts) == self.image_embeds.shape[0]
        self.processor = processor

    def __len__(self):
        return len(self.true_texts)

    def __getitem__(self, idx):
        gt_inputs = self.processor(text=[self.true_texts[idx]], images=None, return_tensors="pt", padding="max_length",  max_length=30, truncation=True)
        cf_inputs = self.processor(text=[self.cf_texts[idx]], images=None, return_tensors="pt", padding="max_length",  max_length=30, truncation=True)
        target_obj_inputs = self.processor(text=[self.target_obj_texts[idx]], images=None, return_tensors="pt", padding="max_length",  max_length=10, truncation=True)
        
        return gt_inputs, cf_inputs, self.image_embeds[idx], target_obj_inputs, torch.tensor(self.gt_label_idx[idx]).long(), torch.tensor(self.cf_label_idx[idx]).long()
    

    def create_dataset(self,data_path,ratio=1):
        data = torch.load(data_path)
        if ratio < 1:
            labels = [item["gt_count"] for item in data]
            data, _, _, _ = train_test_split(data, labels, test_size=1-ratio, random_state=42, stratify=labels)

        true_texts, cf_texts, image_embeds, target_obj_texts, gt_label_idx, cf_label_idx = [], [], [], [], [], []
        for sample in data:
            sample_cf_texts = sample["target_obj_aug_with_context_text"].copy()
            sample_cf_texts.pop(sample["gt_count"]-2)
            cf_texts += sample_cf_texts
            true_texts += [sample["target_obj_aug_with_context_text"][sample["gt_count"]-2]] * len(sample_cf_texts)
            image_embeds += [sample["image_embeds"].detach()]*len(sample_cf_texts)
            target_obj_texts += [sample["target_obj_text"]]*len(sample_cf_texts)
            gt_label_idx += [sample["gt_count"]-2]*len(sample_cf_texts)
            sample_cf_label_idx = np.arange(0,self.num_classes).tolist()
            # try:
            sample_cf_label_idx.pop(sample["gt_count"]-2)
            # except:
            #     print("IndexError: pop index out of range")
            #     print(sample_cf_label_idx,sample["gt_count"])
            #     exit()
            cf_label_idx += sample_cf_label_idx

        return true_texts, cf_texts, torch.cat(image_embeds, dim=0).detach().clone(), target_obj_texts, gt_label_idx, cf_label_idx

class TextDatasetOnlineTrainContrastive(Dataset):
    def __init__(self, data_path, processor, num_classes=4, ratio=1,add_object_cf=False,ref_obj_file=None):
        self.num_classes = num_classes

        self.true_texts, self.cf_texts, self.image_embeds, self.target_obj_texts, self.gt_label_idx, self.cf_label_idx = self.create_dataset(data_path,ratio)

        assert len(self.true_texts) == len(self.cf_texts) == self.image_embeds.shape[0]
        self.processor = processor

    def __len__(self):
        return len(self.true_texts)

    def __getitem__(self, idx):
        gt_inputs = self.processor(text=self.true_texts[idx], images=None, return_tensors="pt", padding="max_length",  max_length=30, truncation=True)
        cf_inputs = self.processor(text=self.cf_texts[idx], images=None, return_tensors="pt", padding="max_length",  max_length=30, truncation=True)
        target_obj_inputs = self.processor(text=self.target_obj_texts[idx], images=None, return_tensors="pt", padding="max_length",  max_length=10, truncation=True)

        # torch.Size([8, 30]), torch.Size([8, 30]), torch.Size([8, 512])
        return gt_inputs, cf_inputs, self.image_embeds[idx], target_obj_inputs, torch.tensor(self.gt_label_idx[idx]).long(), torch.tensor(self.cf_label_idx[idx]).long()


    def create_dataset(self,data_path,ratio=1):
        data = torch.load(data_path)
        if ratio < 1:
            labels = [item["gt_count"] for item in data]
            data, _, _, _ = train_test_split(data, labels, test_size=1-ratio, random_state=42, stratify=labels)

        true_texts, cf_texts, image_embeds, target_obj_texts, gt_label_idx, cf_label_idx = [], [], [], [], [], []

        for sample in data:
            sample_cf_texts = sample["target_obj_aug_with_context_text"].copy()
            sample_cf_texts.pop(sample["gt_count"]-2)
            cf_texts.append(sample_cf_texts)
            true_texts.append(sample["target_obj_aug_with_context_text"][sample["gt_count"]-2])
            image_embeds.append(sample["image_embeds"].detach().unsqueeze(0))
            target_obj_texts.append(sample["target_obj_text"])
            gt_label_idx.append(sample["gt_count"]-2)
            sample_cf_label_idx = np.arange(0,self.num_classes).tolist()
            # try:
            sample_cf_label_idx.pop(sample["gt_count"]-2)
            # except:
            #     print("IndexError: pop index out of range")
            #     print(sample_cf_label_idx,sample["gt_count"])
            #     exit()
            cf_label_idx.append(sample_cf_label_idx)

        # print(image_embeds[0].shape)
        return true_texts, cf_texts, torch.cat(image_embeds, dim=0).detach().clone(), target_obj_texts, gt_label_idx, cf_label_idx


class TextDatasetCustomDogs(Dataset):
    def __init__(self, data_path, ref, processor, clip_model, device, add_object_cf=False,ref_obj_file=None):
    
        self.true_texts, self.cf_texts, self.image_embeds = self.create_dataset_4class(
            data_path, ref, processor, clip_model, device
        )

        assert len(self.true_texts) == len(self.cf_texts) == self.image_embeds.shape[0]
        self.processor = processor
        self.device = device

    def __len__(self):
        return len(self.true_texts)

    def __getitem__(self, idx):
        inputs = self.processor(text=[self.true_texts[idx]], images=None, return_tensors="pt", padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        cf_inputs = self.processor(text=[self.cf_texts[idx]], images=None, return_tensors="pt", padding=True)
        cf_inputs = {k:v.to(self.device) for k,v in cf_inputs.items()}
        return inputs, cf_inputs, self.image_embeds[idx]
    
    def create_dataset_4class(self,data_path, ref, processor, clip_model, device):
        with torch.no_grad():
            augmented_data = data_augmentation({ref:torch.load(data_path)})[ref]

            # true_texts, cf_texts, image_embeds = [], [], []
            number_words = ["two", "three", "four", "five"]
            true_texts, cf_texts, image_embeds, target_obj_texts, gt_label_idx, cf_label_idx = [], [], [], [], [], []
            for idx, number_word in enumerate(number_words):
                for sample in augmented_data[idx+2]:
                    pixel_values=processor(text=None, images=sample["img"], return_tensors="pt", padding=True)["pixel_values"].to(device) # torch.Size([1, 3, 224, 224])
                    image_embeds+=[get_image_embeds(clip_model,pixel_values.to(device),device=device).detach()]*3
                    true_texts += [f"{number_word} {ref}"]*3 # torch.Size([1, 512])
                    number_cf = number_words.copy()
                    number_cf.pop(idx)
                    cf_texts += [f"{ele} {ref}" for ele in number_cf]# torch.Size([3, 512])
                    target_obj_texts += [f"{ref}"]*3
                    gt_label_idx += [idx]*3
                    sample_cf_label_idx = [0,1,2,3]
                    sample_cf_label_idx.pop(idx)
                    cf_label_idx += sample_cf_label_idx

            return true_texts, cf_texts, torch.cat(image_embeds, dim=0).detach().clone(), target_obj_texts, gt_label_idx, cf_label_idx


class TextEmbeddingDataset(Dataset):
    def __init__(self, data_path, ref, processor, clip_model, device, add_object_cf=False,ref_obj_file=None):
    
        self.true_text_embeds, self.cf_text_embeds, self.image_embeds = self.create_dataset_4class(
            data_path, ref, processor, clip_model, device, add_object_cf,ref_obj_file
        )
        self.add_object_cf = add_object_cf

        assert self.true_text_embeds.shape[0] == self.cf_text_embeds.shape[0] == self.image_embeds.shape[0]

    def __len__(self):
        return len(self.true_text_embeds)

    def __getitem__(self, idx):
        return self.true_text_embeds[idx], self.cf_text_embeds[idx], self.image_embeds[idx]
    
    def create_dataset_4class(self,data_path, ref, processor, clip_model, device, add_object_cf,ref_obj_file):
        if add_object_cf:
            with open(ref_obj_file, 'r') as file:
                other_ref_objs = file.readlines()
            other_ref_objs = [line.strip() for line in other_ref_objs]
            if ref in other_ref_objs:
                other_ref_objs.remove(ref)
            random.seed(42)
            true_number = 6
        else:
            true_number = 3
        
        with torch.no_grad():
            # augmented_data = data_augmentation({ref:torch.load(data_path)})[ref]
            # print("Using augmented data")
            
            augmented_data = torch.load(data_path)
            print("Not using augmented data")
            true_text_embeds, cf_text_embeds, image_embeds = [], [], []
            number_words = ["two", "three", "four", "five"]
            for idx, number_word in enumerate(number_words):
                for sample in augmented_data[idx+2]:
                    pixel_values=processor(text=None, images=sample["img"], return_tensors="pt", padding=True)["pixel_values"].to(device) # torch.Size([1, 3, 224, 224])
                    image_embeds+=[get_image_embeds(clip_model,pixel_values.to(device),device=device).detach()]*true_number
                    inputs = processor(text=[f"{number_word} {ref}"], images=None, return_tensors="pt", padding=True)
                    true_text_embeds +=[clip_model.text_model(
                        input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                        position_ids=None,
                        output_attentions=clip_model.config.output_attentions,
                        output_hidden_states=clip_model.config.output_hidden_states,
                        return_dict=clip_model.config.use_return_dict,
                    )[1].detach()]*true_number # torch.Size([1, 512])
                    number_cf = number_words.copy()
                    number_cf.pop(idx)
                    inputs = processor(text=[f"{ele} {ref}" for ele in number_cf], images=None, return_tensors="pt", padding=True)
                    
                    cf_text_embeds.append(
                        clip_model.text_model(
                            input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device),
                            position_ids=None,
                            output_attentions=clip_model.config.output_attentions,
                            output_hidden_states=clip_model.config.output_hidden_states,
                            return_dict=clip_model.config.use_return_dict,
                        )[1].detach()
                    )                                       
                    if add_object_cf:
                        inputs = processor(text=[f"{number_word} {obj}" for obj in random.sample(other_ref_objs, 3)], images=None, return_tensors="pt", padding=True)
                        cf_text_embeds.append(
                            clip_model.text_model(
                                input_ids=inputs["input_ids"].to(device),
                                attention_mask=inputs["attention_mask"].to(device),
                                position_ids=None,
                                output_attentions=clip_model.config.output_attentions,
                                output_hidden_states=clip_model.config.output_hidden_states,
                                return_dict=clip_model.config.use_return_dict,
                            )[1].detach()
                        )                            

            return torch.cat(true_text_embeds, dim=0).detach().clone(), torch.cat(cf_text_embeds, dim=0).detach().clone(), torch.cat(image_embeds, dim=0).detach().clone()

class ProcessedCountBenchDataset(Dataset):
    def __init__(
            self, 
            data,
            device="cuda",
            num_classes=4
    ):
        self.data = data
        self.device = device
        self.num_classes=num_classes
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return sample["image_embeds"],sample['target_obj_text'], sample["target_obj_aug_text"][:self.num_classes], sample["target_obj_with_context_text"], sample["target_obj_aug_with_context_text"][:self.num_classes], sample["gt_count"]


class CustomDataset(Dataset):
    def __init__(
            self, 
            data,
            processor,
            model,
            target_obj,
            number_words,
            device="cuda",
    ):
        self.data = []
        for number in range(2,6):
            for sample in data[number]:
                try:
                    pixel_values=processor(text=None, images=sample["img"], return_tensors="pt", padding=True)["pixel_values"] # torch.Size([1, 3, 224, 224])
                    image_embeds = get_image_embeds(
                        model=model,
                        pixel_values=pixel_values.to(device),
                        device=device
                    ).detach().cpu()
                    self.data.append({
                        "gt_count":number,
                        "target_obj_text":target_obj,
                        "target_obj_aug_text":[f"{number_word} {target_obj}" for number_word in number_words],
                        "image_embeds":image_embeds,
                    }) 
                except:
                    pass
        print("========len(self.data)",len(self.data))

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return sample["image_embeds"],sample['target_obj_text'], sample["target_obj_aug_text"], sample['target_obj_text'], sample["target_obj_aug_text"], sample["gt_count"]

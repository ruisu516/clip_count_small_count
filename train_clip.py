import torch, os, json, random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler, Subset
from transformers import CLIPProcessor, CLIPModel
import argparse
from data_aug import data_augmentation
from my_datasets import TextDatasetCustomDogs,TextDatasetOnlineTrain

import torch
import torch.nn as nn
from transformers import CLIPTextModel
import wandb
import numpy as np
from modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

class CustomCLIPModel(nn.Module):
    def __init__(self, clip_config, clip_text_model, clip_text_projection,number_shift_vectors=None,prefix_embedding=None,prefix_length=None,orthogonalize=True,ablation_proj_to_target_aug_embeds=False,ablation_add_to_target_embeds=False):
        super(CustomCLIPModel, self).__init__()
        self.clip_text_model = clip_text_model
        self.clip_text_projection = clip_text_projection
        self.number_shift_vectors = number_shift_vectors
        self.prefix_embedding = prefix_embedding
        self.prefix_length = prefix_length
        self.orthogonalize = orthogonalize
        self.clip_config = clip_config
        self.ablation_proj_to_target_aug_embeds = ablation_proj_to_target_aug_embeds
        self.ablation_add_to_target_embeds = ablation_add_to_target_embeds
    
    def projection_helper(self,op1,op2):
        return torch.matmul(op1,op2.permute(1,0)).unsqueeze(-1) / torch.sum(op1*op1,dim=1).unsqueeze(-1).unsqueeze(-1) * op1.unsqueeze(1).repeat(1,op2.shape[0],1) 
    
    def forward(
            self, 
            gt_input_ids, 
            true_attention_mask, 
            gt_label_idx, #(bz,num_classes-1)
            cf_input_ids, 
            cf_attention_mask, 
            cf_label_idx, #(bz,num_classes-1)
            target_input_ids=None, 
            target_attention_mask=None,
    ):

        assert cf_label_idx.shape == gt_label_idx.shape
        cf_label_idx,gt_label_idx = cf_label_idx.unsqueeze(-1), gt_label_idx.unsqueeze(-1)
        cf_batch_indices = torch.arange(gt_input_ids.shape[0]).unsqueeze(1).expand_as(cf_label_idx)

        if self.prefix_embedding is not None:
            def _prefix_forward(input_ids,label_idx):
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])

                hidden_states = self.clip_text_model.embeddings(input_ids=input_ids, position_ids=None)

                # expanded_prefix = self.prefix_embedding.unsqueeze(0).expand(input_shape[0], -1, -1)
                # expanded_prefix = self.prefix_embedding[cf_batch_indices,gt_label_idx].squeeze()
                expanded_prefix = self.prefix_embedding[label_idx].squeeze()
                # print("label_idx.shape",label_idx.shape)
                # print("expanded_prefix.shape",expanded_prefix.shape)
                # print("hidden_states.shape",hidden_states.shape)

                hidden_states = torch.cat([expanded_prefix, hidden_states], dim=1)

                # CLIP's text model uses causal mask, prepare it here.
                # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
                extended_input_shape = (input_shape[0], input_shape[1] + self.prefix_length)  # (batch_size, prefix + sequence_length)
                causal_attention_mask = _create_4d_causal_attention_mask(
                    extended_input_shape, hidden_states.dtype, device=device
                )


                encoder_outputs = self.clip_text_model.encoder(
                    inputs_embeds=hidden_states,
                    attention_mask=None,
                    causal_attention_mask=causal_attention_mask,
                    output_attentions=self.clip_text_model.config.output_attentions,
                    output_hidden_states=self.clip_text_model.config.output_hidden_states,
                    return_dict=self.clip_text_model.config.use_return_dict,
                )


                last_hidden_state = self.clip_text_model.final_layer_norm(encoder_outputs[0])

                if self.clip_text_model.eos_token_id == 2:
                    # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
                    # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
                    # ------------------------------------------------------------
                    # text_embeds.shape = [batch_size, sequence_length, transformer.width]
                    # take features from the eot embedding (eot_token is the highest number in each sequence)
                    # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
                    pooled_output = last_hidden_state[
                        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                        input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
                    ]
                else:
                    # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
                    pooled_output = last_hidden_state[
                        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                        # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                        # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                        (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.clip_text_model.eos_token_id)
                        .int()
                        .argmax(dim=-1),
                    ]
                return pooled_output
            
            true_text_embeds = self.clip_text_projection(_prefix_forward(gt_input_ids,gt_label_idx))
            cf_text_embeds = self.clip_text_projection(_prefix_forward(cf_input_ids,cf_label_idx))  

        else:
            true_text_embeds = self.clip_text_projection(
                self.clip_text_model(
                    input_ids=gt_input_ids,
                    attention_mask=true_attention_mask,
                    position_ids=None,
                    output_attentions=self.clip_config.output_attentions,
                    output_hidden_states=self.clip_config.output_hidden_states,
                    return_dict=self.clip_config.use_return_dict,
                )[1]
            )#(bz,dim)
            cf_text_embeds = self.clip_text_projection(
                self.clip_text_model(
                    input_ids=cf_input_ids,
                    attention_mask=cf_attention_mask,
                    position_ids=None,
                    output_attentions=self.clip_config.output_attentions,
                    output_hidden_states=self.clip_config.output_hidden_states,
                    return_dict=self.clip_config.use_return_dict,
                )[1]
            ) #(bz,dim)

            if self.number_shift_vectors is not None:
                target_embeds = self.clip_text_projection(
                            self.clip_text_model(
                                input_ids=target_input_ids,
                                attention_mask=target_attention_mask,
                                position_ids=None,
                                output_attentions=self.clip_config.output_attentions,
                                output_hidden_states=self.clip_config.output_hidden_states,
                                return_dict=self.clip_config.use_return_dict,
                            )[1]
                        ) # (bz, dim)
                
                if self.orthogonalize:
                    if self.ablation_proj_to_target_aug_embeds:
                        gt_projection = self.projection_helper(true_text_embeds,self.number_shift_vectors)
                        gt_number_shift_vectors = self.number_shift_vectors.unsqueeze(0) - gt_projection #(bz,num_classes,dim)
                        cf_projection = self.projection_helper(cf_text_embeds,self.number_shift_vectors)
                        cf_number_shift_vectors = self.number_shift_vectors.unsqueeze(0) - cf_projection #(bz,num_classes,dim)
                    
                    else:
                        #projection (num_classes,bz, dim)
                        projection = self.projection_helper(target_embeds,self.number_shift_vectors)
                        gt_number_shift_vectors = self.number_shift_vectors.unsqueeze(0) - projection #(bz,num_classes,dim)
                        cf_number_shift_vectors = gt_number_shift_vectors
                else:
                    gt_number_shift_vectors = self.number_shift_vectors.unsqueeze(0).repeat(true_text_embeds.shape[0],1,1) #(bz,num_classes,dim)
                    cf_number_shift_vectors = gt_number_shift_vectors
                
                
                
                # print(cf_batch_indices.shape,number_shift_vectors.shape,gt_label_idx.shape)
                # print(number_shift_vectors[cf_batch_indices,gt_label_idx].squeeze().shape,true_text_embeds.shape)
                if self.ablation_add_to_target_embeds:
                    true_text_embeds = target_embeds + gt_number_shift_vectors[cf_batch_indices,gt_label_idx].squeeze()
                    cf_text_embeds = target_embeds + cf_number_shift_vectors[cf_batch_indices,cf_label_idx].squeeze()
                else:
                    true_text_embeds = true_text_embeds + gt_number_shift_vectors[cf_batch_indices,gt_label_idx].squeeze()
                    cf_text_embeds = cf_text_embeds + cf_number_shift_vectors[cf_batch_indices,cf_label_idx].squeeze()
                # true_text_embeds = true_text_embeds + number_shift_vectors[gt_label_idx]
                # cf_text_embeds = cf_text_embeds + number_shift_vectors[cf_label_idx]

                    
        text_embeds = torch.concat([true_text_embeds, cf_text_embeds], dim=0) # (2*bz,dim)
        return text_embeds / torch.norm(text_embeds, p=2, dim=-1, keepdim=True)

def save_metadata(args, filename='metadata.json'):
    """Save the parsed arguments to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(args, f, indent=4)

def get_image_embeds(model,pixel_values,device="cpu"):
    vision_outputs = model.vision_model(
            pixel_values=pixel_values.to(device),
            output_attentions=model.config.output_attentions,
            output_hidden_states=model.config.output_hidden_states,
            return_dict=model.config.use_return_dict,
        )
    image_embeds = vision_outputs[1]
    image_embeds = model.visual_projection(image_embeds)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

    return image_embeds

def count_loss(logits_per_true_text, logits_per_cf_text,):
    # Extract the diagonal elements
    true_diag = torch.diag(logits_per_true_text)
    cf_diag = torch.diag(logits_per_cf_text)
    count_loss = torch.mean(torch.log(1 + torch.exp(cf_diag - true_diag)))

    if self.args.add_object_cf:
        # Step 2: Create a mask for the negative pairs
        eye = torch.eye(logits_per_true_text.size(0)).bool().to(self.device)
        negative_mask = ~eye  # Invert the identity matrix to get the negative mask
        row_negative_logits = logits_per_true_text.masked_select(negative_mask).view(logits_per_true_text.size(0), -1) #(bz,bz-1)
        row_positive_logits = true_diag.unsqueeze(1) #(bz,1)
        row_obj_loss = torch.mean(torch.log(1 + torch.exp(row_negative_logits - row_positive_logits)))
        
        col_negative_logits = logits_per_true_text.t().masked_select(negative_mask).view(logits_per_true_text.size(0), -1) #(bz,bz-1)
        col_positive_logits = true_diag.unsqueeze(1) #(bz,1)
        col_obj_loss = torch.mean(torch.log(1 + torch.exp(col_negative_logits - col_positive_logits)))
        
        return count_loss + 0.5 * row_obj_loss + 0.5 * col_obj_loss
    else:
        return count_loss

def count_loss(logits_per_true_text, logits_per_cf_text):
    # Extract the diagonal elements
    true_diag = torch.diag(logits_per_true_text)
    cf_diag = torch.diag(logits_per_cf_text)
    count_loss = torch.mean(torch.log(1 + torch.exp(cf_diag - true_diag)))

    if self.args.add_object_cf:
        # Step 2: Create a mask for the negative pairs
        eye = torch.eye(logits_per_true_text.size(0)).bool().to(self.device)
        negative_mask = ~eye  # Invert the identity matrix to get the negative mask
        row_negative_logits = logits_per_true_text.masked_select(negative_mask).view(logits_per_true_text.size(0), -1) #(bz,bz-1)
        row_positive_logits = true_diag.unsqueeze(1) #(bz,1)
        row_obj_loss = torch.mean(torch.log(1 + torch.exp(row_negative_logits - row_positive_logits)))
        
        col_negative_logits = logits_per_true_text.t().masked_select(negative_mask).view(logits_per_true_text.size(0), -1) #(bz,bz-1)
        col_positive_logits = true_diag.unsqueeze(1) #(bz,1)
        col_obj_loss = torch.mean(torch.log(1 + torch.exp(col_negative_logits - col_positive_logits)))
        
        return count_loss + 0.5 * row_obj_loss + 0.5 * col_obj_loss
    else:
        return count_loss

class Trainer:
    def __init__(self, dataloader, val_dataloader, clip_model, logit_scale, model_save_path, args,device="cuda"):
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.args = args
        self.logit_scale = logit_scale
        self.train_number_shift_vectors = args.train_number_shift_vectors
        self.prefix_tunning = args.prefix_tunning
        self.prefix_length = args.prefix_length
        self.clip_config = clip_model.config
        self.device=device
               
        trainable_parameters = []
        for name,param in clip_model.named_parameters():
            if any([ele in name for ele in self.args.trainable_parameters]):
                param.requires_grad = True
                trainable_parameters.append(param)
            else:
                param.requires_grad = False
        
        if self.train_number_shift_vectors:
                       # number_shift_vectors.requires_grad = True
            if args.number_shift_vectors_init_weight_path is not None:
                print("loading number_shift_vectors from",args.number_shift_vectors_init_weight_path)
                loaded_tensor = torch.load(args.number_shift_vectors_init_weight_path).to(device)
                loaded_tensor.requires_grad_(True)
                number_shift_vectors = torch.nn.Parameter(loaded_tensor)
                # number_shift_vectors = torch.nn.Parameter(torch.load(args.number_shift_vectors_init_weight_path).to(device))
            else:
                torch.manual_seed(42)
                number_shift_vectors = torch.nn.Parameter(torch.randn(args.num_classes, self.clip_config.projection_dim).to(device))

            # number_shift_vectors = number_shift_vectors.to(device)
            trainable_parameters.append(number_shift_vectors)
        else:
            number_shift_vectors = None
        if self.prefix_tunning:
            prefix_embedding = torch.nn.Parameter(torch.randn(args.num_classes, args.prefix_length, self.clip_config.text_config.hidden_size).to(device))
            trainable_parameters.append(prefix_embedding)
        else:
            prefix_embedding = None
        
        self.text_model = CustomCLIPModel(self.clip_config,clip_model.text_model.to(device), clip_model.text_projection.to(device), \
                                          number_shift_vectors,prefix_embedding,args.prefix_length,args.orthogonalize,\
                                            ablation_proj_to_target_aug_embeds=self.args.ablation_proj_to_target_aug_embeds,\
                                                ablation_add_to_target_embeds=self.args.ablation_add_to_target_embeds)

        print("len(trainable_parameters)",len(trainable_parameters))
        if self.args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(trainable_parameters, lr=self.args.lr)
        elif self.args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(trainable_parameters, lr=self.args.lr)
        
        self.training_losses = {}
        self.validation_losses = {}
        self.trained_epochs = 0
        self.model_save_path = model_save_path

    # def count_loss(self, logits_per_true_text, logits_per_cf_text):
    #     # return -torch.mean(torch.log(torch.exp(logits_per_true_text) / (torch.exp(logits_per_true_text) + torch.exp(logits_per_cf_text))))
    #     return torch.mean(torch.log(1+torch.exp(logits_per_cf_text-logits_per_true_text)))
        
    def count_loss(self, logits_per_true_text, logits_per_cf_text):
        # Extract the diagonal elements
        true_diag = torch.diag(logits_per_true_text)
        cf_diag = torch.diag(logits_per_cf_text)
        count_loss = torch.mean(torch.log(1 + torch.exp(cf_diag - true_diag)))

        if self.args.add_object_cf:
            # Step 2: Create a mask for the negative pairs
            eye = torch.eye(logits_per_true_text.size(0)).bool().to(self.device)
            negative_mask = ~eye  # Invert the identity matrix to get the negative mask
            row_negative_logits = logits_per_true_text.masked_select(negative_mask).view(logits_per_true_text.size(0), -1) #(bz,bz-1)
            row_positive_logits = true_diag.unsqueeze(1) #(bz,1)
            row_obj_loss = torch.mean(torch.log(1 + torch.exp(row_negative_logits - row_positive_logits)))
            
            col_negative_logits = logits_per_true_text.t().masked_select(negative_mask).view(logits_per_true_text.size(0), -1) #(bz,bz-1)
            col_positive_logits = true_diag.unsqueeze(1) #(bz,1)
            col_obj_loss = torch.mean(torch.log(1 + torch.exp(col_negative_logits - col_positive_logits)))
            
            return count_loss + 0.5 * row_obj_loss + 0.5 * col_obj_loss
        else:
            return count_loss
    
    def val(self,eval_data_mode="val"):
        self.text_model.eval()

        if eval_data_mode=="val":
            loader = self.val_dataloader
        elif eval_data_mode=="train":
            loader = self.dataloader

        with torch.no_grad():  # No need to track gradients
            total_loss = 0
            total_samples = 0
            for gt_inputs, cf_inputs, image_embeds, target_obj_inputs, gt_label_idx, cf_label_idx in loader:
                bs = image_embeds.shape[0]

                text_embeds = self.text_model(
                    gt_input_ids=gt_inputs["input_ids"].squeeze().to(self.device), 
                    true_attention_mask=gt_inputs["attention_mask"].squeeze().to(self.device), 
                    gt_label_idx=gt_label_idx.to(self.device),
                    cf_input_ids=cf_inputs["input_ids"].squeeze().to(self.device), 
                    cf_attention_mask=cf_inputs["attention_mask"].squeeze().to(self.device), 
                    cf_label_idx=cf_label_idx.to(self.device),
                    target_input_ids=target_obj_inputs["input_ids"].squeeze().to(self.device), 
                    target_attention_mask=target_obj_inputs["attention_mask"].squeeze().to(self.device),
                )

                logits_per_text = torch.matmul(text_embeds, image_embeds.detach().t().to(self.device)) * self.logit_scale
                logits_per_true_text, logits_per_cf_text = logits_per_text[:bs], logits_per_text[bs:]

                loss = self.count_loss(logits_per_true_text, logits_per_cf_text)
                total_loss += loss.item() * bs
                total_samples += bs

            avg_val_loss = total_loss / total_samples
            # print(f'Validation Loss: {avg_val_loss}')

        self.text_model.train()  # Set model back to training mode
        return avg_val_loss
    
    def save_loss_logs(self):
        loss_log = {
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }
        with open(os.path.join(self.model_save_path, f'loss_log_lr{self.args.lr}.json'), 'w') as f:
            json.dump(loss_log, f, indent=4)

    def train(self, max_num_epochs):
        self.text_model.train()

        avg_val_loss = self.val(eval_data_mode="val")  # Capture validation loss for the current epoch
        pbar = tqdm(total=max_num_epochs, desc=f'Epoch 0/{max_num_epochs}, Training Loss: N/A, Val Loss: {avg_val_loss:.4f}')
        
        if not self.args.not_log_wandb:
            wandb.log({
                f"train_loss":None,
                f"val_loss":avg_val_loss,
                f"trained_epochs":0, 
            })
            
        best_val_loss,best_ep = avg_val_loss,0
        early_stopped = False
        for epoch in range(max_num_epochs):
            cumulative_train_loss = 0
            counter = 0
            for gt_inputs, cf_inputs, image_embeds, target_obj_inputs, gt_label_idx, cf_label_idx in self.dataloader:
                bs = image_embeds.shape[0]
                self.text_model.zero_grad()

                text_embeds = self.text_model(
                    gt_input_ids=gt_inputs["input_ids"].squeeze().to(self.device), 
                    true_attention_mask=gt_inputs["attention_mask"].squeeze().to(self.device), 
                    gt_label_idx=gt_label_idx.to(self.device),
                    cf_input_ids=cf_inputs["input_ids"].squeeze().to(self.device), 
                    cf_attention_mask=cf_inputs["attention_mask"].squeeze().to(self.device), 
                    cf_label_idx=cf_label_idx.to(self.device),
                    target_input_ids=target_obj_inputs["input_ids"].squeeze().to(self.device), 
                    target_attention_mask=target_obj_inputs["attention_mask"].squeeze().to(self.device),
                )

                logits_per_text = torch.matmul(text_embeds, image_embeds.detach().t().to(self.device)) * self.logit_scale
                logits_per_true_text, logits_per_cf_text = logits_per_text[:bs], logits_per_text[bs:]

                loss = self.count_loss(logits_per_true_text, logits_per_cf_text)
                cumulative_train_loss += (loss.detach().item() * bs)
                counter += bs

                loss.backward()
                self.optimizer.step()
            
            self.trained_epochs += 1
            avg_val_loss = self.val(eval_data_mode="val")  # Capture validation loss for the current epoch
            self.training_losses[self.trained_epochs] = cumulative_train_loss / counter
            self.validation_losses[self.trained_epochs] = avg_val_loss

            if avg_val_loss < best_val_loss:
                model_path = os.path.join(self.model_save_path, f'best_model.pth')
                best_val_loss = avg_val_loss
                best_ep = self.trained_epochs
                torch.save(self.text_model.state_dict(), model_path)

            if not self.args.not_log_wandb:
                wandb.log({
                    f"train_loss":cumulative_train_loss / counter,
                    f"val_loss":avg_val_loss,
                    f"trained_epochs":self.trained_epochs,
                })

            # Saving the model after each epoch
            pbar.set_description(f'Epoch {epoch + 1}/{max_num_epochs}, Training Loss: {cumulative_train_loss / counter}, Val Loss: {avg_val_loss}')
            pbar.update(1)

            if self.trained_epochs - best_ep > 10 and best_ep > 0:
                early_stopped = True
                print("Early stopping")
                break

        self.save_loss_logs()
        if not self.args.not_log_wandb:
            wandb.config.best_ep = best_ep
            wandb.config.best_val_loss = best_val_loss
            wandb.config.early_stopping = early_stopped
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on customized dataset")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--ref_obj",type=str,default=None,help="name of the object being used as an reference")   
    parser.add_argument("--optimizer",type=str,default="SGD",choices=["SGD","Adam"])   
    parser.add_argument("--train_data_path",type=str,help="path to custom data")
    parser.add_argument("--eval_data_path",type=str,help="path to custom data")
    parser.add_argument("--model",type=str,choices=["openai/clip-vit-base-patch32","openai/clip-vit-base-patch16","openai/clip-vit-large-patch14","stable_diffusion"],help="choose the model")
    parser.add_argument('--trainable_parameters', nargs='+', help='List of trainable parameters',default=[],choices=["text_model","text_projection"])
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=1)
    parser.add_argument("--random_seed", type=int, default=None)



    parser.add_argument("--number_shift_vectors_init_weight_path",type=str,default=None)
    parser.add_argument("--save_root_folder",type=str)
    parser.add_argument('--add_object_cf', action='store_true')
    parser.add_argument('--CLIP_loss', type=str,default="",choices=["both",""])
    parser.add_argument('--train_number_shift_vectors', action='store_true')
    parser.add_argument('--orthogonalize', action='store_true',help="orthogonalize the number shift vectors w.r.t. the target object") 
    parser.add_argument("--ref_obj_file",type=str,default=None,help="path to the ref objects")   
    parser.add_argument('--not_log_wandb', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ablation_proj_to_target_aug_embeds', action='store_true')
    parser.add_argument('--ablation_add_to_target_embeds', action='store_true')

    parser.add_argument('--prefix_tunning', action='store_true')
    parser.add_argument("--prefix_length", type=int, default=1)


    args = parser.parse_args()
    args.num_epochs = int(args.num_epochs/args.train_ratio)
    print("args.num_epochs",args.num_epochs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained(args.model)
    clip_model = CLIPModel.from_pretrained(args.model).to(device)

    for name,param in clip_model.named_parameters():
        param.requires_grad = False
    if args.ref_obj == "dogs":
        dataset = TextDatasetCustomDogs(args.train_data_path, args.ref_obj, processor, clip_model, device, add_object_cf=args.add_object_cf,ref_obj_file=args.ref_obj_file)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # val_dataset = TextDatasetCustomDogs(args.eval_data_path, args.ref_obj, processor, clip_model, device, add_object_cf=args.add_object_cf,ref_obj_file=args.ref_obj_file)
        # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.ref_obj is None:
        dataset = TextDatasetOnlineTrain(args.train_data_path,processor,num_classes=args.num_classes,ratio=args.train_ratio)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = TextDatasetOnlineTrain(args.eval_data_path, processor,num_classes=args.num_classes)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


    print("len(dataset),len(val_dataset)",len(dataset),len(val_dataset))
    logit_scale = clip_model.logit_scale.exp()
    
    print("trainable_parameters",args.trainable_parameters)
    trained_param_save_name = "_".join(args.trainable_parameters)
    model_save_path = f"{args.save_root_folder}/{args.model.split('/')[1]}_{trained_param_save_name}{args.train_number_shift_vectors}{args.orthogonalize}{args.ablation_proj_to_target_aug_embeds}{args.ablation_add_to_target_embeds}_{args.prefix_tunning}{args.prefix_length}_{args.lr}_{args.optimizer}{args.ref_obj}_{args.add_object_cf}{args.CLIP_loss}"
    os.makedirs(model_save_path, exist_ok=True)
    print("@@@@@@@@@@@@@@@@@@@@save_path:",model_save_path)


    if not args.not_log_wandb:
        wandb.init(project="train_clip_count", config=args.__dict__, entity="ruisu")

    trainer = Trainer(
        dataloader, 
        val_dataloader, 
        clip_model, 
        logit_scale, 
        model_save_path,
        args,
    )

    trainer.train(args.num_epochs)
    args_dict = vars(args)
    
    # Save the dictionary as a JSON file
    save_metadata(args_dict,filename=os.path.join(model_save_path, 'metadata.json'))


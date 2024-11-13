import torch, os, json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import argparse
from data_aug import data_augmentation

"""
train the whole CLIP text encoder
not finished 
"""

class TextDataset(Dataset):
    def __init__(self, data_path, ref, processor, clip_model, device):
    
        self.true_texts, self.cf_texts, self.image_embeds = self.create_dataset_two2five(
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
    
    def create_dataset_two2five(self,data_path, ref, processor, clip_model, device):
        with torch.no_grad():
            augmented_data = data_augmentation({ref:torch.load(data_path)})[ref]

            true_texts, cf_texts, image_embeds = [], [], []
            number_words = ["two", "three", "four", "five"]
            for idx, number_word in enumerate(number_words):
                for sample in augmented_data[idx+2]:
                    pixel_values=processor(text=None, images=sample["img"], return_tensors="pt", padding=True)["pixel_values"].to(device) # torch.Size([1, 3, 224, 224])
                    image_embeds+=[get_image_embeds(clip_model,pixel_values.to(device),device=device).detach()]*3
                    true_texts += [f"{number_word} {ref}"]*3 # torch.Size([1, 512])
                    number_cf = number_words.copy()
                    number_cf.pop(idx)
                    cf_texts += [f"{ele} {ref}" for ele in number_cf]# torch.Size([3, 512])

            return true_texts, cf_texts, torch.cat(image_embeds, dim=0).detach().clone()

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

class Trainer:
    def __init__(self, dataloader, val_dataloader, text_model, text_projection_module, lr, logit_scale, model_save_path,optimizer_name):
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.lr = lr
        self.logit_scale = logit_scale

        train_params = []
        self.text_projection_module = text_projection_module
        self.text_projection_module.requires_grad_(True)
        for p in self.text_projection_module.parameters():
            train_params.append(p)
        self.text_model = text_model
        self.text_model.requires_grad_(True)
        for p in self.text_model.parameters():
            train_params.append(p)

        if optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(train_params, lr=self.lr)
        elif optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(train_params, lr=self.lr)
        self.optimizer_name = optimizer_name
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
        
        return torch.mean(torch.log(1 + torch.exp(cf_diag - true_diag)))

    def val(self,eval_data_mode="val"):
        if eval_data_mode=="val":
            loader = self.val_dataloader
        elif eval_data_mode=="train":
            loader = self.dataloader
        self.text_projection_module.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No need to track gradients
            total_loss = 0
            total_samples = 0
            for true_text_embeds, cf_text_embeds, image_embeds in loader:
                bs = true_text_embeds.shape[0]
                text_embeds = self.text_projection_module(torch.concat([true_text_embeds.detach(), cf_text_embeds.detach()], dim=0))
                text_embeds = text_embeds / torch.norm(text_embeds, p=2, dim=1, keepdim=True)

                logits_true_text = torch.matmul(text_embeds, image_embeds.detach().t()) * self.logit_scale
                logits_per_true_text, logits_per_cf_text = logits_true_text[:bs], logits_true_text[bs:]

                loss = self.count_loss(logits_per_true_text, logits_per_cf_text)
                total_loss += loss.item() * bs
                total_samples += bs

            avg_val_loss = total_loss / total_samples
            # print(f'Validation Loss: {avg_val_loss}')

        self.text_projection_module.train()  # Set model back to training mode
        return avg_val_loss
    
    def save_loss_logs(self):
        loss_log = {
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }
        with open(os.path.join(self.model_save_path, f'loss_log_lr{self.lr}.json'), 'w') as f:
            json.dump(loss_log, f, indent=4)

    def train(self, max_num_epochs):
        self.text_model.train()
        self.text_projection_module.train()
        
        avg_val_loss = self.val(eval_data_mode="val")  # Capture validation loss for the current epoch
        pbar = tqdm(total=max_num_epochs, desc=f'Epoch 0/{max_num_epochs}, Training Loss: N/A, Val Loss: {avg_val_loss:.4f}')
        
        for epoch in range(max_num_epochs):
            cumulative_train_loss = 0
            counter = 0
            for true_text_inputs, cf_text_onputs, image_embeds in self.dataloader:
                bs = true_text_inputs.shape[0]
                self.text_model.zero_grad()
                self.text_projection_module.zero_grad()

                true_text_embeds = self.text_model(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                    position_ids=None,
                    output_attentions=clip_model.config.output_attentions,
                    output_hidden_states=clip_model.config.output_hidden_states,
                    return_dict=clip_model.config.use_return_dict,
                )
                # cf_text_embeds = 
                
                text_embeds = self.text_projection_module(torch.concat([true_text_embeds.detach(), cf_text_embeds.detach()], dim=0))
                text_embeds = text_embeds / torch.norm(text_embeds, p=2, dim=1, keepdim=True)

                logits_true_text = torch.matmul(text_embeds, image_embeds.detach().t()) * self.logit_scale
                logits_per_true_text, logits_per_cf_text = logits_true_text[:bs], logits_true_text[bs:]

                loss = self.count_loss(logits_per_true_text, logits_per_cf_text)
                cumulative_train_loss += (loss.detach().item() * bs)
                counter += bs

                loss.backward()
                self.optimizer.step()
            
            self.trained_epochs += 1
            avg_val_loss = self.val(eval_data_mode="val")  # Capture validation loss for the current epoch
            self.training_losses[self.trained_epochs] = cumulative_train_loss / counter
            self.validation_losses[self.trained_epochs] = avg_val_loss

            # Saving the model after each epoch
            torch.save(self.text_projection_module.state_dict(), os.path.join(self.model_save_path, f'model_epoch_{self.trained_epochs}.pth'))
            print(f'Epoch {epoch + 1}/{max_num_epochs}, Training Loss: {cumulative_train_loss / counter}, Val Loss: {avg_val_loss}')
            # pbar.set_description(f'Epoch {epoch + 1}/{max_num_epochs}, Training Loss: {cumulative_train_loss / counter:.4f}, Val Loss: {avg_val_loss:.4f}')
            pbar.update(1)

        self.save_loss_logs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on customized dataset")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--ref_obj",type=str,default="dogs",help="name of the object being used as an reference")   
    parser.add_argument("--optimizer",type=str,default="SGD",choices=["SGD","Adam"])   
    parser.add_argument("--train_data_path",type=str,help="path to custom data")
    parser.add_argument("--eval_data_path",type=str,help="path to custom data")
    parser.add_argument("--model",type=str,choices=["openai/clip-vit-base-patch32","openai/clip-vit-base-patch16","openai/clip-vit-large-patch14","stable_diffusion"],help="choose the model")
    parser.add_argument("--model_save_path",type=str,help="path to custom data")
    parser.add_argument('--train_whole_text_encoder', action='store_true')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained(args.model)
    clip_model = CLIPModel.from_pretrained(args.model).to(device)
    for name,param in clip_model.named_parameters():
        param.requires_grad = False

    dataset = TextEmbeddingDataset(args.train_data_path, args.ref_obj, processor, clip_model, device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = TextEmbeddingDataset(args.eval_data_path, args.ref_obj, processor, clip_model, device)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print("len(dataset),len(val_dataset)",len(dataset),len(val_dataset))
    logit_scale = clip_model.logit_scale.exp()

    my_clip_text_projection = torch.nn.Linear(clip_model.text_projection.in_features,clip_model.text_projection.out_features,bias=False)
    my_clip_text_projection.weight.data = clip_model.text_projection.weight.data.detach().clone()
    my_clip_text_projection = my_clip_text_projection.to(device)
    
    if not os.path.isdir(args.model_save_path):
        os.mkdir(args.model_save_path)
    trainer = Trainer(
        dataloader, 
        val_dataloader, 
        my_clip_text_projection, 
        args.lr, 
        logit_scale, 
        args.model_save_path,
        args.optimizer,
    )
    trainer.train(args.num_epochs)


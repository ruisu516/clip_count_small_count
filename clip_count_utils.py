import torch, spacy, os
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm
from requests.exceptions import Timeout
from modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

spacy_nlp = spacy.load("en_core_web_sm")

OBJ_NAMES = {
    'dogs':"dogs", 
    'lions':"lions", 
    'chairs':"chairs", 
    'laptops':"laptops",
    'cats':"cats", 
    'home_cat':"cats", 
    'outside_cats':"cats", 
    'cartoon_cats':"cats",
    'goats':'goats',
    'cows':'cows', 
    'cherries':'cherries', 
    'roses':'roses', 
    'boats':'boats',
}
NUMBER_WORDS_SUB = [
        "two", "three", "four", "five",
    ]
NUMBER_WORDS = [
    "two", "three", "four", "five",
    "six", "seven", "eight", "nine",
    'ten'
]
ARABIC_NUMBER_WORDS = ["2","3","4","5","6","7","8","9","10"]
SUB_NUMBER_RANGE=[2,3,4,5]

def generate_random_vectors(shape, seed,N=10):
    torch.manual_seed(seed)
    vectors = []
    
    for _ in range(N):
        vec = torch.randn(*shape)
        vectors.append(vec)
        
    concatenated_vectors = torch.stack(vectors)
    return concatenated_vectors

def project_tensor_B_onto_A(A, B):

    # Ensure the tensors have the correct shape
    assert A.shape == B.shape, "Tensors must have same shape"

    # Initialize the resulting tensor
    proj_B_on_A = torch.zeros_like(A)

    # Project each vector of B onto A
    for i in range(A.shape[1]):
        a = A[0,i,:]
        b = B[0,i,:]

        dot_product_b_a = torch.dot(b, a)
        dot_product_a_a = torch.dot(a, a)

        projection = (dot_product_b_a / dot_product_a_a) * a
        proj_B_on_A[0,i,:] = projection

    return proj_B_on_A

def get_prompt_embeds(model,input_ids,attention_mask,normalize=True):
    text_outputs = model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                output_attentions=model.config.output_attentions,
                output_hidden_states=model.config.output_hidden_states,
                return_dict=model.config.use_return_dict,
            )
    text_embeds = text_outputs[1] #torch.Size([9, 512])
    if hasattr(model, "text_projection"):
        text_embeds = model.text_projection(text_embeds) #torch.Size([9, 512])
    if normalize:
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    return text_embeds

def text2embedding(text,model,processor,device,normalize):
    inputs = processor(text=text, images=None, return_tensors="pt", padding=True)
    if inputs["input_ids"].shape[1] > 77:
        last_id = inputs["input_ids"][:,[-1]]
        inputs["input_ids"] = torch.concat([inputs["input_ids"][:,:76],last_id],dim=1)
        # last_mask = inputs["attention_mask"][:,[-1]]
        # inputs["attention_mask"] = torch.concat([inputs["attention_mask"][:,:76],last_mask],dim=1)
    return get_prompt_embeds(
        model=model,
        input_ids=inputs["input_ids"].to(device),
        attention_mask=None, #inputs["attention_mask"].to(device),
        normalize=normalize,
    )# torch.Size([k, 512])

def text2embedding_with_prefix_embedding(text,model,processor,prefix_embedding,prefix_length,device,normalize):
    def _prefix_forward():
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = model.text_model.embeddings(input_ids=input_ids, position_ids=None)

        # expanded_prefix = prefix_embedding.unsqueeze(0).expand(input_shape[0], -1, -1)
        print("prefix_embedding.shape",prefix_embedding.shape)
        print("hidden_states.shape",prefix_embedding.shape)

        hidden_states = torch.cat([prefix_embedding, hidden_states], dim=1)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        extended_input_shape = (input_shape[0], input_shape[1] + prefix_length)  # (batch_size, prefix + sequence_length)
        causal_attention_mask = _create_4d_causal_attention_mask(
            extended_input_shape, hidden_states.dtype, device=device
        )


        encoder_outputs = model.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            causal_attention_mask=causal_attention_mask,
            output_attentions=model.text_model.config.output_attentions,
            output_hidden_states=model.text_model.config.output_hidden_states,
            return_dict=model.text_model.config.use_return_dict,
        )


        last_hidden_state = model.text_model.final_layer_norm(encoder_outputs[0])

        if model.text_model.eos_token_id == 2:
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
        text_embeds = model.text_projection(pooled_output)
        if normalize:
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds
    
    inputs = processor(text=text, images=None, return_tensors="pt", padding=True)
    if inputs["input_ids"].shape[1] > 77:
        last_id = inputs["input_ids"][:,[-1]]
        inputs["input_ids"] = torch.concat([inputs["input_ids"][:,:76],last_id],dim=1)    
    
    return _prefix_forward()# torch.Size([k, 512])

def get_target_rep(target_object,target_aug_sentences,model,processor,device="cuda",normalize=True):
    target_object_prompt_embeds = text2embedding(target_object,model,processor,device,normalize)
    target_aug_text_embeds = text2embedding(target_aug_sentences,model,processor,device,normalize)

    return target_object_prompt_embeds,target_aug_text_embeds


def get_ref_difference(ref_aug_sentences,ref_object,model,processor,device="cuda",normalize=True,batch_first=True):
    ref_prompt_multi = text2embedding(ref_aug_sentences,model,processor,device,normalize) # (bz*num_classes,
    ref_prompt_single = text2embedding(ref_object,model,processor,device,normalize)[:,None,:]  # 1
    if batch_first:
        batch_size = len(ref_object)
        num_classes = int(len(ref_aug_sentences)/batch_size)
        ref_prompt_multi = ref_prompt_multi[np.arange(batch_size * num_classes).reshape(num_classes,batch_size).T.flatten()].reshape(batch_size, num_classes, -1)
        ref_diff=ref_prompt_multi-ref_prompt_single
    else:
        ref_diff=ref_prompt_multi.reshape(-1,ref_prompt_single.shape[0],ref_prompt_multi.shape[-1]).permute(1,0,2)-ref_prompt_single
    return ref_diff,ref_prompt_single

def apply_reff_diff(start,end,ref_diff,factor,linear_shift,start_with_target_with_num):
        if linear_shift:
            if not start_with_target_with_num:
                merged_text_embeds=factor*(start+ref_diff)+(1-factor)*end
            else: 
                merged_text_embeds = end + factor * ref_diff
        else:
            raise NotImplementedError
        return merged_text_embeds

def get_image_embeds(model,pixel_values,device="cuda"):
    vision_outputs = model.vision_model(
            pixel_values=pixel_values.to(device),
            output_attentions=model.config.output_attentions,
            output_hidden_states=model.config.output_hidden_states,
            return_dict=model.config.use_return_dict,
        )
    image_embeds = vision_outputs[1]
    if hasattr(model, "visual_projection"):
        image_embeds = model.visual_projection(image_embeds)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

    return image_embeds

def get_logits(model,text_embeds,image_embeds):
    # print(model.device,text_embeds.device,image_embeds.device)
    if len(text_embeds.shape) == 2:
        text_embeds = text_embeds[None,...]
    if len(image_embeds.shape) == 2:
        image_embeds = image_embeds[None,...]
    logit_scale = model.logit_scale.exp()
    logits_per_text = torch.bmm(text_embeds, image_embeds.permute(0,2,1)) * logit_scale

    return logits_per_text,logits_per_text.permute(0,2,1)


def run_on_my_data_img_retrievel(
        model,
        processor,
        target_data,
        target,
        ref,
        normalize,
        device,
        factor,
        linear_shift=True,
        start_with_target_with_num=True,
):
    ref_aug_sentences=[f"{word} {ref}" for word in NUMBER_WORDS[:num_classes]]
    target_aug_sentences=[f"{word} {OBJ_NAMES[target]}" for word in NUMBER_WORDS[:num_classes]]

    ref_diff,ref_prompt_single=get_ref_difference(
        ref_aug_sentences=ref_aug_sentences,
        ref_object=[ref],
        model=model,
        processor=processor,
        device=device,
        normalize=normalize
    )
    target_text_sample={}
    target_text_sample["target_obj_embeds"],target_text_sample["target_obj_aug_embeds"]=get_target_rep(
        target_object=[OBJ_NAMES[target]],
        target_aug_sentences=target_aug_sentences,
        model=model,
        processor=processor,
        device=device,
        normalize=normalize
    )
    ref_diff_projection = (torch.bmm(ref_diff, ref_prompt_single.permute(0,2,1)) / torch.bmm(ref_prompt_single, ref_prompt_single.permute(0,2,1)).squeeze()) * ref_prompt_single
    ref_diff_projection_2 = (torch.sum(ref_diff-ref_diff_projection * target_text_sample["target_obj_aug_embeds"], dim=1, keepdim=True)/torch.sum(target_text_sample["target_obj_aug_embeds"] * target_text_sample["target_obj_aug_embeds"], dim=1, keepdim=True))*target_text_sample["target_obj_aug_embeds"]
    ref_diff = ref_diff - ref_diff_projection - ref_diff_projection_2 
    merged_text_embeds = apply_reff_diff(target_text_sample["target_obj_embeds"],target_text_sample["target_obj_aug_embeds"],ref_diff,factor,linear_shift,start_with_target_with_num)
        
    # if normalize_before_scoring:
    merged_text_embeds=merged_text_embeds/merged_text_embeds.norm(p=2,dim=-1,keepdim=True)

    all_probs_per_class=[]
    for text_num in range(2,num_classes+2):
        all_logits_per_text = []
        merged_text_embeds_ = merged_text_embeds[text_num-2]
        for number in range(2,num_classes+2):
            
            # print(merged_text_embeds_.size())
            selected_data = target_data[number]
            for sample in selected_data:
                pixel_values=processor(text=None, images=sample["img"], return_tensors="pt", padding=True)["pixel_values"] # torch.Size([1, 3, 224, 224])
                image_embeds = get_image_embeds(
                    model=model,
                    pixel_values=pixel_values.to(device),
                    device=device
                )
                logits_per_text,_= get_logits(model,merged_text_embeds_,image_embeds)
                all_logits_per_text.append(logits_per_text.item())
            torch.cuda.empty_cache()

        all_probs_per_class.append(torch.nn.functional.softmax(torch.tensor(all_logits_per_text).float(),dim=0).reshape(num_classes,-1).sum(dim=1).numpy().tolist())
    
    return all_probs_per_class

def get_ref_diff_helper(model,processor,ref,target_text_sample,device,normalize,num_classes):
    ref_aug_sentences=[f"{word} {ref}" for word in NUMBER_WORDS[:num_classes]]

    # TODO: get_ref_difference() is changed to return tensors of (bz,num_classes,emb_dim)
    ref_diff,ref_prompt_single=get_ref_difference(
        ref_aug_sentences=ref_aug_sentences,
        ref_object=[ref],
        model=model,
        processor=processor,
        device=device,
        normalize=normalize
    )
    
    ref_diff_projection = (torch.bmm(ref_diff, ref_prompt_single.permute(0,2,1)) / torch.bmm(ref_prompt_single, ref_prompt_single.permute(0,2,1)).squeeze()) * ref_prompt_single
    ref_diff_projection_2 = (torch.sum((ref_diff-ref_diff_projection) * target_text_sample["target_obj_embeds"], dim=1, keepdim=True)/torch.sum(target_text_sample["target_obj_embeds"] * target_text_sample["target_obj_embeds"], dim=1, keepdim=True))*target_text_sample["target_obj_embeds"]
    ref_diff = ref_diff - ref_diff_projection - ref_diff_projection_2 #+ (1-factor) * (tar_diff - tar_diff_projection - tar_diff_aligned)
    
    return ref_diff

def get_ref_diff_multi_objs(model,processor,target_text_sample,device,normalize,num_classes,ref_objs):
    ref_diff = 0
    for ref in ref_objs:
        ref_diff += get_ref_diff_helper(model,processor,ref,target_text_sample,device,normalize,num_classes)
    ref_diff /= len(ref_objs)
    return ref_diff

def run_on_my_data_clf(
        model,
        processor,
        target_data,
        target,
        ref,
        factor=1,
        normalize=False,
        device='cuda',
        num_classes=4,
        linear_shift=True,
        start_with_target_with_num=True,
        use_only_number_word=False,
        normalize_number_word=None,
        use_random_vector=False,
        random_seed=None,
        use_muti_objs=False,
):
    target_text_sample={}
    target_aug_sentences=[f"{word} {OBJ_NAMES[target]}" for word in NUMBER_WORDS[:num_classes]]
    target_text_sample["target_obj_embeds"],target_text_sample["target_obj_aug_embeds"]=get_target_rep(
        target_object=[OBJ_NAMES[target]],
        target_aug_sentences=target_aug_sentences,
        model=model,
        processor=processor,
        device=device,
        normalize=normalize
    )
    # print("target_text_sample['target_obj_embeds'].shape,",target_text_sample["target_obj_embeds"].shape,)
    if ref is None:
        print("ref if none, set merged_text_embeds=target_obj_aug_embeds")
        merged_text_embeds = target_text_sample["target_obj_aug_embeds"]
    else:
        if use_random_vector:
            print("use_random_vector")
            ref_diff_obj = get_ref_diff_helper(model,processor,ref,target_text_sample,device,normalize,num_classes)
            torch.manual_seed(random_seed)
            ref_diff = torch.randn(*ref_diff_obj.shape).to(device)
            ref_diff_projection_2 = (torch.bmm(ref_diff, target_text_sample["target_obj_embeds"].permute(0,2,1)) / torch.bmm(target_text_sample["target_obj_embeds"], target_text_sample["target_obj_embeds"].permute(0,2,1)).squeeze()) * target_text_sample["target_obj_embeds"]
            ref_diff = ref_diff - ref_diff_projection_2 #+ (1-factor) * (tar_diff - tar_diff_projection - tar_diff_aligned)
            ref_diff = ref_diff * ref_diff_obj.norm(p=2,dim=-1,keepdim=True) / ref_diff.norm(p=2,dim=-1,keepdim=True)

        elif use_only_number_word:
            print("use_only_number_word")
            ref_aug_sentences=[f"{word}" for word in NUMBER_WORDS[:num_classes]]
            ref_diff = text2embedding(ref_aug_sentences,model,processor,device,True)
            ref_diff_projection_2 = (torch.bmm(ref_diff, target_text_sample["target_obj_embeds"].permute(0,2,1)) / torch.bmm(target_text_sample["target_obj_embeds"], target_text_sample["target_obj_embeds"].permute(0,2,1)).squeeze()) * target_text_sample["target_obj_embeds"]
            ref_diff = ref_diff - ref_diff_projection_2 #+ (1-factor) * (tar_diff - tar_diff_projection - tar_diff_aligned)
            if normalize_number_word:
                # TODO: the norm should be taken on normalized ref_diff
                ref_diff_norm = get_ref_diff_helper(model,processor,ref,target_text_sample,device,normalize,num_classes).norm(p=2,dim=-1,keepdim=True) 
                ref_diff = ref_diff * ref_diff_norm / ref_diff.norm(p=2,dim=-1,keepdim=True)

        elif use_muti_objs:
            print("use_muti_objs")
            ref_diff = get_ref_diff_multi_objs(model,processor,target_text_sample,device,normalize,num_classes,ref)
        else:
            ref_diff = get_ref_diff_helper(model,processor,ref,target_text_sample,device,normalize,num_classes)

        merged_text_embeds = apply_reff_diff(target_text_sample["target_obj_embeds"],target_text_sample["target_obj_aug_embeds"],ref_diff,factor,linear_shift,start_with_target_with_num)

    merged_text_embeds=merged_text_embeds/merged_text_embeds.norm(p=2,dim=-1,keepdim=True)

    flat_predictions=[]
    flat_labels=[]
    for number in range(2,num_classes+2):
        selected_data = target_data[number]
        predictions=[]
        for sample in selected_data:
            pixel_values=processor(text=None, images=sample["img"], return_tensors="pt", padding=True)["pixel_values"] # torch.Size([1, 3, 224, 224])
            image_embeds = get_image_embeds(
                model=model,
                pixel_values=pixel_values.to(device),
                device=device
            )
            _,logits_per_image= get_logits(model,merged_text_embeds,image_embeds)
            probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label prob
            predictions.append(torch.argmax(probs).item()+2)
        flat_predictions+=predictions
        flat_labels+=[number]*len(predictions)

    acc=np.mean(np.array(flat_predictions)==np.array(flat_labels)).round(4)*100

    return flat_predictions,flat_labels,acc

def run_countbench_sample(
        sample,
        model,
        processor,
        normalize=False,
        factor=1,
        num_classes=4,
        ref_obj=None,
        linear_shift=True,
        device="cuda",
        use_ref_with_context=True, # TODO: check this part
        start_with_target_with_num=True,
        use_target_obj_with_context=True,
        use_target_aug_sent_with_context=True
):
    if not use_target_obj_with_context:
        start=sample["target_obj_embeds"].to(device)
    else:
        start=sample["target_obj_embeds_with_context"].to(device)
    if not use_target_aug_sent_with_context:
        end=sample["target_obj_aug_embeds"][:num_classes].to(device)
    else:
        end=sample["target_obj_aug_embeds_with_context"][:num_classes].to(device)
    
    if factor == 0:
        merged_text_embeds=end
    else:
        if use_ref_with_context:
            ref_diff_per_sample,ref_prompt_single = get_ref_difference(
                ref_aug_sentences=[ele.replace(sample["target_obj"],ref_obj) for ele in sample["target_obj_aug_with_context"]][:num_classes],
                ref_object=sample["target_obj_with_context"].replace(sample["target_obj"],ref_obj),
                model=model,
                processor=processor,
                device=device,
                normalize=normalize
            )
        else:
            ref_diff_per_sample,ref_prompt_single= get_ref_difference(
                ref_aug_sentences=[f"{number} {ref_obj}" for number in NUMBER_WORDS[:num_classes]],
                ref_object=[ref_obj],
                model=model,
                processor=processor,
                device=device,
                normalize=normalize
            )
        ref_diff_projection = (torch.bmm(ref_diff_per_sample, ref_prompt_single.permute(0,2,1)) / torch.bmm(ref_prompt_single, ref_prompt_single.permute(0,2,1)).squeeze()) * ref_prompt_single
        ref_diff_aligned_B = (torch.sum(ref_diff_per_sample-ref_diff_projection * end, dim=1, keepdim=True)/torch.sum(end * end, dim=1, keepdim=True))*end

        ref_diff_per_sample = ref_diff_per_sample - ref_diff_projection - ref_diff_aligned_B #+ (1-factor) * (tar_diff - tar_diff_projection - tar_diff_aligned)
        merged_text_embeds = apply_reff_diff(start,end,ref_diff_per_sample,factor,linear_shift,start_with_target_with_num)
        
    merged_text_embeds=merged_text_embeds/merged_text_embeds.norm(p=2,dim=-1,keepdim=True)
    merged_text_embeds=merged_text_embeds[:num_classes]
    return merged_text_embeds

def contains_word(sentence, word_list):
    words = sentence.lower().split()
    for word in word_list:
        if word.lower() in words:
            return True, word
    return False, None


    
def sentence_augmentation(sentence):
    sentence=sentence.lower()

    new_sentences = []
    obj_with_nums = []
    object_name = ""
    sentence_no_num=None

    contains,word=contains_word(sentence, NUMBER_WORDS)
    if contains:
        doc = spacy_nlp(sentence.split(word)[1])
        for np_ in doc.noun_chunks:
            if np_.text.strip() is not None:
                object_name = np_.text.strip()
            break
        if object_name!="":
            if (sentence.split(word)[1].split(object_name)[0].strip()!=""):
                object_name=sentence.split(word)[1].split(object_name)[0].strip()+" "+object_name
        for number_word in NUMBER_WORDS:
            new_sentences.append(sentence.replace(word, number_word))
            obj_with_nums.append(f"{number_word} {object_name}")
        sentence_no_num=sentence.replace(f"{word} ","")

    return new_sentences,object_name,sentence_no_num,obj_with_nums


def countbench_streaming_data(sample,model,processor,device="cuda",number=None,normalize=True):
    if (number is not None) and (sample["number"] != number):
        return None
    if sample["image"] is None:
        try:
            image = Image.open(requests.get(sample["image_url"], stream=True,timeout=2).raw)
        except Timeout:
            print("timeout")
            return None
        except:
            # print(f"Error loading {sample['image_url']}")
            return None
    else:
        image = sample["image"]
    target_obj_aug_with_context,target_obj,target_obj_with_context,target_obj_aug = sentence_augmentation(sample["text"])

    target_obj_embeds,target_obj_aug_embeds=get_target_rep(
        target_object=target_obj,
        target_aug_sentences=target_obj_aug,
        model=model,
        processor=processor,
        normalize=normalize,
        device=device,
    )
    target_obj_embeds_with_context,target_obj_aug_embeds_with_context=get_target_rep(
        target_object=target_obj_with_context,
        target_aug_sentences=target_obj_aug_with_context,
        model=model,
        processor=processor,
        normalize=normalize,
        device=device,
    )

    pixel_values=processor(text=None, images=image, return_tensors="pt", padding=True)["pixel_values"] # torch.Size([1, 3, 224, 224])
    image_embeds = get_image_embeds(
        model=model,
        pixel_values=pixel_values,
        device=device,
    ) 

    return {
            "number":sample['number'],
            "target_obj":target_obj,
            "target_obj_embeds":target_obj_embeds,
            "target_obj_aug":target_obj_aug,
            "target_obj_aug_embeds":target_obj_aug_embeds,
            "target_obj_with_context":target_obj_with_context,
            "target_obj_embeds_with_context":target_obj_embeds_with_context,
            "target_obj_aug_with_context":target_obj_aug_with_context,
            "target_obj_aug_embeds_with_context":target_obj_aug_embeds_with_context,
            "image_embeds":image_embeds,
        }


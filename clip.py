from clip_count_utils import *
import os
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm
from my_datasets import *
import pickle
import wandb


DESIRED_COLUMNS = ["average", "dogs", "cats", "lions", "chairs", "goats", "cows", "cherries", "roses", "boats", "ref"]

# model_name= "openai/clip-vit-base-patch32" # "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32" "openai/clip-vit-base-patch16"
# model_name="openai/clip-vit-large-patch14"
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# device="cuda"
# model = CLIPModel.from_pretrained(model_name).to(device)
# model.requires_grad=False
# processor = CLIPProcessor.from_pretrained(model_name)


# def get_file_name(model_name,ref,target_obj,data_name="",num_classes="",extension="csv"):
#     if ref is None:
#         ref_str = "None"
#     elif isinstance(ref,str):
#         ref_str = ref
#     elif isinstance(ref,list):
#         ref_str = f"{len(ref)}_refs"
#     target_name = target_obj if target_obj is not None else "None"
#     return f"{model_name.split('/')[1]}_{ref_str}_{target_name}_{data_name}_{num_classes}.{extension}"


def image_retrievel(model_name,ref_obj,sample_size,augmented_data,eval_dir,device="cpu"):
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.requires_grad=False
    processor = CLIPProcessor.from_pretrained(model_name)

    normalize=False
    # sample_size = len(augmented_data['dogs'][2])
    num_classes = 4
    linear_shift=True
    task = 'image_retrievel'
    start_with_target_with_num = True


    all_probs_by_factors = []
    all_mean_probs_by_factors = []
    all_probs_by_target = []
    for factor in [0,1]: # output original results and results after applying our method
        # all_probs_by_target = []
        all_mean_probs_by_target = []
        for target in augmented_data.keys():
            all_probs_per_class=run_on_my_data_img_retrievel(
                model=model,
                processor=processor,
                target_data= augmented_data[target],
                target=target,
                ref=ref_obj,
                normalize=normalize,
                device=device,
                factor=factor,
                sample_size=sample_size,
                num_classes=num_classes,
                linear_shift=linear_shift,
                start_with_target_with_num=start_with_target_with_num)
            mean_prob = np.mean([all_probs_per_class[i][i] for i in range(len(all_probs_per_class))])
            all_mean_probs_by_target.append(mean_prob)
            # all_probs_by_target.append(all_probs_per_class)
        # all_probs_by_factors.append(all_probs_by_target)
        all_mean_probs_by_factors.append(all_mean_probs_by_target)

    # pb_pd = pd.DataFrame(all_probs_by_factors,columns=list(augmented_data.keys()))
    # pb_pd.index = args.factors_list[1:]
    # pb_pd.to_csv(f"csv/final/{fn}")

    mean_pb_pd = pd.DataFrame(all_mean_probs_by_factors,columns=list(augmented_data.keys()))
    mean_pb_pd.index = [[ele]*len(all_mean_probs_by_target) for ele in [0,1]]
    mean_pb_pd["average"] = np.array(all_mean_probs_by_factors).mean(axis=1)
    mean_pb_pd.to_csv(os.path.join(eval_dir,get_file_name(task,model_name,ref_obj,data_name="custom_data",num_classes=num_classes)))

def generate_random_vectors(shape, seed,N=10):
    torch.manual_seed(seed)
    vectors = []
    
    for _ in range(N):
        vec = torch.randn(*shape)
        vectors.append(vec)
        
    concatenated_vectors = torch.stack(vectors)
    return concatenated_vectors



# def img_clf_custom(
#         model_name,
#         model,
#         processor,
#         ref_objs,
#         target_data,
#         eval_dir,
#         args,
#         task_name="img_clf_custom",
#         device="gpu",
# ):
#     """
#         run image classification on custom dataset
#     """
#     if args.use_arabic_nums:
#         number_words = ARABIC_NUMBER_WORDS[:args.num_classes]
#     else:
#         number_words = NUMBER_WORDS[:args.num_classes]


#     print(f"Running task {task_name}, using model {model_name}")
#     # Initialize an empty list to hold accuracy by ref to build up the DataFrame incrementally
    

#     # Check if the evaluation directory exists, if not, create it
#     if not os.path.isdir(eval_dir):
#         os.mkdir(eval_dir)

#     # Initialize or clear the file
#     file_name = get_file_name(task_name, model_name, ref_objs, data_name="", num_classes="", factor=args.factor)
#     file_path = os.path.join(eval_dir, file_name)
#     # Create an empty DataFrame or clear the existing file to start fresh
#     pd.DataFrame().to_csv(file_path)

#     if not args.use_multi_objs:
#         iteration = ref_objs 
#     else:
#         iteration = [ref_objs]
#     acc_by_ref = []
#     for ref in tqdm(iteration, desc=f"Processing refs for task {task_name}"):
#         acc_by_target = []
#         # Adding a progress bar for iterating over target_data keys
#         print(f"Processing targets for ref: {ref}")
#         for target in tqdm(target_data.keys()):
#             _, _, acc = run_on_my_data_clf(
#                 model=model,
#                 processor=processor,
#                 target_data=target_data[target],
#                 target=target,
#                 ref=ref,
#                 factor=args.factor,
#                 normalize=normalize,
#                 device=device,
#                 num_classes=4,
#                 linear_shift=linear_shift,
#                 start_with_target_with_num=start_with_target_with_num,
#                 use_only_number_word=args.use_only_number_word,
#                 normalize_number_word=args.normalize_number_word,
#                 use_random_vector=use_random_vector,
#                 random_seed=random_seed,
#                 use_multi_objs=args.use_multi_objs,
#             )
#             acc_by_target.append(acc)
#         acc_by_ref.append(acc_by_target)

#         # Update the DataFrame and save after processing each `ref`
#         acc_pd = pd.DataFrame(np.array(acc_by_ref), columns=list(target_data.keys()))
#         acc_pd["average"] = np.array(acc_by_ref).mean(axis=1)
#         acc_pd["ref"] = iteration[:len(acc_by_ref)]  # Match the length of acc_by_ref to avoid index out of bounds
#         # Save/update the CSV file
#         print(f"update {file_path}...")
#         try:
#             acc_pd = acc_pd[DESIRED_COLUMNS]
#         except:
#             pass
#         acc_pd.to_csv(file_path, index=False)  # Use index=False to avoid writing row indices

def generate_file_path(args,ext="pth"):
    """
    Generates a unique file path based on the provided arguments.
    Transforms all arguments into a string, replaces slashes with underscores,
    and uses a hash to ensure the filename is unique and not excessively long.
    """
    # Construct a list of argument values, replacing any slashes in strings
    components = []
    for key, value in sorted(args.items()):
        if key in ["root_folder","custom_data_path","processed_countbench_data_path","test_batch_size","random_seed","not_log_wandb","trained_clip_path"]:
            continue
        if "train_" in key:
            continue
        if value is None:
            continue
        value = str(value).replace("/", "_")
        components.append(f"{value}")

    path_string = '_'.join(components)

    return f"{path_string}.{ext}"

def img_clf(
    target_data,
    model,
    processor,
    args,
    number_shift_vectors=None,
    prefix_embedding=None,
    prefix_length=None,
    target_obj=None,
    ref_obj=None,
    normalize=False,
    linear_shift=True,
    start_with_target_with_num=True,
    device="cuda",
):
    def get_ref_embed_helper(ref_obj):

        if args.use_ref_with_context:
            ref_object = [context.replace(org,ref_obj) for org,context in zip(target_obj_text,target_obj_with_context_text)]
            ref_aug_sentences = [item.replace(target,ref_obj) for tuple_ in target_obj_aug_with_context_text for item,target in zip(tuple_,target_obj_text)]

        else:
            ref_object = [ref_obj]*len(target_obj_text)
            ref_aug_sentences=[]
            for number in number_words:
                ref_aug_sentences += [f"{number} {ref_obj}"]*batch_size
        
        return get_ref_difference(
            ref_aug_sentences=ref_aug_sentences,
            ref_object=ref_object,
            model=model,
            processor=processor,
            device=device,
            normalize=normalize,
            batch_first=True
        )
    
    def proj_1_helper(op1,op2,disable_ref_orth,device):
        if disable_ref_orth:
            return torch.zeros_like(op1).to(device)
        else:
            return (torch.bmm(op1, op2.permute(0,2,1)) / torch.bmm(op2, op2.permute(0,2,1))) * op2

    def proj_2_helper(op1,op2,disable_target_orth,device):
        """
            op1: (bz, num_class, embed_dim)
            op2: (bz, embed_dim, 1)
        """
        if disable_target_orth:
            return torch.zeros_like(op1).to(device)
        else:
            return torch.bmm(op1, op2)/torch.sum(op2 * op2, dim=1, keepdim=True)*op2.permute(0,2,1).repeat(1,args.num_classes,1)

    def proj_2_helper_ablation(op1,op2,device):
        """
            op1: (bz, num_class, embed_dim)
            op2: (bz, num_class, embed_dim)
        """
        dot_products = torch.sum(op1 * op2, dim=-1, keepdim=True) # (bz, num_class ,1)
        op2_norm = torch.sum(op2 * op2, dim=-1, keepdim=True) # (bz, num_class, 1)

        return dot_products/op2_norm*op2


    def merge_helper_orth_once(ref_diff):
        if args.ablation_proj_to_target_aug_embeds:
            ref_diff_projection_2 = proj_2_helper_ablation(ref_diff,target_aug_embeds,device)
        else:
            ref_diff_projection_2 = proj_2_helper(ref_diff, target_embeds,args.disable_target_orth,device)
        ref_diff = ref_diff - ref_diff_projection_2 #+ (1-args.factor) * (tar_diff - tar_diff_projection - tar_diff_aligned)

        if args.use_only_number_word and args.normalize_number_word and ref_obj is not None:
            obj_ref_diff,obj_ref_prompt_single = get_ref_embed_helper(ref_obj)
            obj_ref_diff_projection = proj_1_helper(obj_ref_diff,obj_ref_prompt_single,args.disable_ref_orth,device)
            if args.ablation_proj_to_target_aug_embeds:
                obj_ref_diff_projection_2 = proj_2_helper_ablation(obj_ref_diff-obj_ref_diff_projection,target_aug_embeds,device)
            else:
                obj_ref_diff_projection_2 = proj_2_helper(obj_ref_diff-obj_ref_diff_projection,target_embeds,args.disable_target_orth,device)
            obj_ref_diff = obj_ref_diff - obj_ref_diff_projection - obj_ref_diff_projection_2 #+ (1-args.factor) * (tar_diff - tar_diff_projection - tar_diff_aligned)

            ref_diff = ref_diff * obj_ref_diff.norm(p=2,dim=-1,keepdim=True) / ref_diff.norm(p=2,dim=-1,keepdim=True)
        
        if args.ablation_add_to_target_embeds:
            # print("target_embeds.shape",target_embeds.shape)
            # print("ref_diff.shape",ref_diff.shape)
            return apply_reff_diff(target_embeds,target_embeds.permute(0,2,1),ref_diff,args.factor,linear_shift,start_with_target_with_num)
        else:
            return apply_reff_diff(target_embeds,target_aug_embeds,ref_diff,args.factor,linear_shift,start_with_target_with_num)
    

    def merge_helper_multi(ref_diff_list,ref_prompt_single_list,semantic_similarity_list):
        orth_ref_diff = 0
        for ref_diff, ref_prompt_single,semantic_similarity in zip(ref_diff_list,ref_prompt_single_list,semantic_similarity_list):
            ref_diff_projection = proj_1_helper(ref_diff,ref_prompt_single,args.disable_ref_orth,device)
            if args.ablation_proj_to_target_aug_embeds:
                ref_diff_projection_2 = proj_2_helper_ablation(ref_diff-ref_diff_projection,target_aug_embeds,device)
            else:
                ref_diff_projection_2 = proj_2_helper(ref_diff-ref_diff_projection, target_embeds,args.disable_target_orth,device)
            orth_ref_diff += (ref_diff - ref_diff_projection - ref_diff_projection_2)*semantic_similarity.view(batch_size,1,1)#+ (1-args.factor) * (tar_diff - tar_diff_projection - tar_diff_aligned)
            del ref_diff,ref_prompt_single,ref_diff_projection,ref_diff_projection_2
        if args.ablation_add_to_target_embeds:
            return apply_reff_diff(target_embeds,target_embeds.permute(0,2,1),orth_ref_diff/len(ref_obj),args.factor,linear_shift,start_with_target_with_num)
        else:
            return apply_reff_diff(target_embeds,target_aug_embeds,orth_ref_diff/len(ref_obj),args.factor,linear_shift,start_with_target_with_num)
    
                
    def merge_helper_orth_twice(ref_diff,ref_prompt_single,semantic_similarity):
        ref_diff_projection = proj_1_helper(ref_diff,ref_prompt_single,args.disable_ref_orth,device)
        if args.ablation_proj_to_target_aug_embeds:
            ref_diff_projection_2 = proj_2_helper_ablation(ref_diff-ref_diff_projection,target_aug_embeds,device)
        else:
            ref_diff_projection_2 = proj_2_helper(ref_diff-ref_diff_projection, target_embeds,args.disable_target_orth,device)

        ref_diff = (ref_diff - ref_diff_projection - ref_diff_projection_2) * semantic_similarity.view(batch_size,1,1) #+ (1-args.factor) * (tar_diff - tar_diff_projection - tar_diff_aligned)
        
        if args.ablation_add_to_target_embeds:
            return apply_reff_diff(target_embeds,target_embeds.permute(0,2,1),ref_diff,args.factor,linear_shift,start_with_target_with_num)
        else:
            return apply_reff_diff(target_embeds,target_aug_embeds,ref_diff,args.factor,linear_shift,start_with_target_with_num)


    configs = args.__dict__
    configs["target_obj"] = target_obj
    configs["normalize"] = normalize
    configs["linear_shift"] = linear_shift
    configs["start_with_target_with_num"] = start_with_target_with_num
    if not args.not_log_wandb:
        wandb.login(key = "f120e5e4c8c84329e87f496f85e6f7ded7732680")
        api = wandb.Api()
        runs = api.runs(path="ruisu/clip_count_new")
        for run in runs:
            config_matches = all(run.config.get(key) == value for key, value in configs.items())
            if config_matches and run.state == "finished":
                print(f"Run {run.id} matches the target configuration and is finished.")
                return None
        
        if args.load_trained_clip:
            try:
                with open(os.path.join(args.trained_clip_path,"metadata.json"), 'r') as file:
                    train_configs = json.load(file)
                for key in train_configs:
                    configs[f"train_{key}"] = train_configs[key]
            except:
                print("Not find metadata")
                pass
        
        wandb.init(project="clip_count_new", config=configs, entity="ruisu")
    print(configs)

    use_target_aug_sent_with_context = not args.not_use_target_aug_sent_with_context

    if args.use_arabic_nums:
        number_words = ARABIC_NUMBER_WORDS[:args.num_classes]
    else:
        number_words = NUMBER_WORDS[:args.num_classes]

    
    if "countbench" in args.dataset or "coco-count" in args.dataset:
        dataset = ProcessedCountBenchDataset(
            data=target_data,
            device=device,
            num_classes=args.num_classes
        )
    elif "custom" in args.dataset:
        dataset = CustomDataset(
            data=target_data,
            processor=processor,
            model=model,
            target_obj=target_obj,
            number_words=number_words,
            device=device,
        )


    print("len(dataset)",len(dataset))
    dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False)

    """
        target_embeds: (bz, embed_dim, 1)
        target_aug_embeds: (bz, num_class, embed_dim)
        ref_diff: (bz, num_class, embed_dim)
        ref_prompt_single: (bz, 1, embed_dim)
        semantic_similarity: (bz)
        semantic_similarity_list: (num_ref_objs,bz)

    """
    
    predictions = []
    gt_labels = []
    
    for image_embeds,target_obj_text,target_obj_aug_text,target_obj_with_context_text,target_obj_aug_with_context_text, gt_count in dataloader:
        batch_size = len(image_embeds)            
                
            
        target_obj_aug_with_context_text = [item for tuple_ in target_obj_aug_with_context_text for item in tuple_]
        target_obj_aug_text = [item for tuple_ in target_obj_aug_text for item in tuple_]

        
        if prefix_embedding is None:
            target_embeds = text2embedding(
                target_obj_with_context_text if args.use_target_obj_with_context else target_obj_text,
                model,processor,device,normalize
            )[...,None]

            target_aug_embeds = text2embedding(
                target_obj_aug_with_context_text if use_target_aug_sent_with_context else target_obj_aug_text,
                model,processor,device,normalize
            )[np.arange(batch_size * args.num_classes).reshape(args.num_classes,batch_size).T.flatten()].reshape(batch_size, args.num_classes, -1)
        else:
            target_aug_embeds = text2embedding_with_prefix_embedding(target_obj_aug_with_context_text if use_target_aug_sent_with_context else target_obj_aug_text,model,processor,prefix_embedding,prefix_length,device,normalize)
        
        
        if ref_obj is not None:
            
            if args.use_multi_objs: # 1.use_multi_objs
                ref_diff_list,ref_prompt_single_list,semantic_similarity_list=[],[],[]
                for r in ref_obj:
                    ref_diff,ref_prompt_single = get_ref_embed_helper(r)
                    ref_diff_list.append(ref_diff)
                    ref_prompt_single_list.append(ref_prompt_single)
                    if args.use_abs_semantic_weight or args.use_normalized_semantic_weight:
                        semantic_similarity = torch.bmm(ref_prompt_single/ref_prompt_single.norm(p=2,dim=-1,keepdim=True),target_embeds/target_embeds.norm(p=2,dim=1,keepdim=True)).squeeze()[None,...]
                    else:   
                        semantic_similarity = torch.ones(1, batch_size).to(device)
                    semantic_similarity_list.append(semantic_similarity)
                semantic_similarity_list = torch.stack(semantic_similarity_list)
                
                if args.use_normalized_semantic_weight:
                    semantic_similarity_list = semantic_similarity_list/semantic_similarity_list.sum(dim=0,keepdim=True)*len(ref_obj)
                    # print("semantic_similarity_list[:,0]",semantic_similarity_list[:,0])        
                merged_text_embeds = merge_helper_multi(ref_diff_list,ref_prompt_single_list,semantic_similarity_list)
            
            else: # 2. single ref obj
                ref_diff,ref_prompt_single = get_ref_embed_helper(ref_obj)
                if args.use_abs_semantic_weight:
                    semantic_similarity = torch.bmm(ref_prompt_single/ref_prompt_single.norm(p=2,dim=-1,keepdim=True),target_embeds/target_embeds.norm(p=2,dim=1,keepdim=True)).squeeze()
                else:   
                    semantic_similarity = torch.ones(batch_size).to(device)
                merged_text_embeds = merge_helper_orth_twice(ref_diff,ref_prompt_single,semantic_similarity)
        else:
            if args.use_trained_linear_vector_as_ref and number_shift_vectors is not None: # 3. trained linear shift vector
                ref_diff = number_shift_vectors.unsqueeze(0).repeat(batch_size, 1, 1)
                semantic_similarity = torch.ones(batch_size).to(device)
                # print(ref_diff.shape)
                merged_text_embeds = merge_helper_orth_once(ref_diff)
            
            elif args.use_only_number_word: # 4. only number words
                ref_aug_sentences=[f"{word}" for word in number_words]
                ref_diff = text2embedding(ref_aug_sentences,model,processor,device,normalize).unsqueeze(0).repeat(batch_size, 1, 1)
                merged_text_embeds = merge_helper_orth_once(ref_diff)
            
            elif args.use_self_as_ref: # 5. self as ref
                ref_embeds = text2embedding(target_obj_text,model,processor,device,normalize)[:,None,:]
                ref_aug_embeds = text2embedding(target_obj_aug_text,model,processor,device,normalize)
                ref_aug_embeds = ref_aug_embeds[np.arange(batch_size * args.num_classes).reshape(args.num_classes,batch_size).T.flatten()].reshape(batch_size, args.num_classes, -1)
                ref_diff = ref_embeds - ref_aug_embeds
                merged_text_embeds = merge_helper_orth_once(ref_diff)
            else: # 6. default
                merged_text_embeds = target_aug_embeds
            
        
        
        merged_text_embeds=merged_text_embeds/merged_text_embeds.norm(p=2,dim=-1,keepdim=True)
        _,logits_per_image= get_logits(model,merged_text_embeds,image_embeds.to(device)) # (bz,1,args.num_classes)
        result = torch.argmax(logits_per_image,dim=-1).squeeze().detach().cpu().numpy()+2
        if isinstance(result, np.ndarray):
            predictions.extend(result.tolist())
        else:
            predictions.append(result)
        gt_labels.extend(gt_count.tolist())
    
    
    acc = np.round((np.array(predictions)==np.array(gt_labels)).mean()*100,2)
    exp_results = {
        "predictions":predictions,
        "gt_labels":gt_labels,
        "acc":float(acc)
    }

    # file_name = get_file_name(args.model, ref_obj, target_obj,data_name=args.dataset, num_classes=args.num_classes, extension="pth")
    # file_name = generate_file_path(configs,ext="pth")
    # save_path = os.path.join(args.root_folder,file_name)
    # print(f"Saving to {save_path}")
    # with open(save_path, 'wb') as f:
    #     pickle.dump(exp_results, f)
    
    if not args.not_log_wandb:
        wandb.log({f"acc":acc})
        wandb.finish()
    
    print(f"Accuracy: {acc}%","Target object:",target_obj)

    return exp_results
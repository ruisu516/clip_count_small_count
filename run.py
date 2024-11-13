
import argparse,torch
from sd import reproduce_stable_diffusion_results
from clip import *
from data_aug import data_augmentation
import wandb
from transformers import AutoProcessor, AutoModel, BlipModel



if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="An example script to parse command-line arguments.")

    # Add arguments
    parser.add_argument("--root_folder",type=str,help="path to custom data")
    parser.add_argument("--dataset",type=str,choices=["custom","countbench","coco-count-val","coco-count-test"],help="choose from custom dataset or countbench")
    parser.add_argument("--custom_data_path",type=str,help="path to custom data")
    parser.add_argument("--coco_eval_data_path",type=str,help="path to custom data")
    parser.add_argument("--processed_countbench_data_path",type=str,help="path to custom data")
    parser.add_argument("--task",type=str,choices=["classification","image_retrievel","image_gen"],help="choose the task")
    
    parser.add_argument("--model",type=str,help="choose the model")
    # parser.add_argument("--model",type=str,choices=["openai/clip-vit-base-patch32","openai/clip-vit-base-patch16","openai/clip-vit-large-patch14","stable_diffusion"],help="choose the model")
    # parser.add_argument("--trained_text_projection_path",default="",type=str)   
    parser.add_argument("--trained_clip_path",default="",type=str)   
    
    parser.add_argument("--test_batch_size",type=int,default=32) 
    parser.add_argument("--factor",type=float,default=1)   
    parser.add_argument("--num_classes",type=int,default=4)   
    parser.add_argument("--random_seed",type=int,default=None)    
    parser.add_argument("--ref_obj",type=str,default=None,help="name of the object being used as an reference")   
    parser.add_argument("--ref_obj_file",type=str,default=None,help="path to the ref objects")   


    # parser.add_argument('--load_trained_text_projection', action='store_true')
    parser.add_argument('--load_trained_clip', action='store_true')
    parser.add_argument("--normalize_number_word",action='store_true')   
    parser.add_argument('--not_use_target_aug_sent_with_context', action='store_true')
    parser.add_argument('--use_abs_semantic_weight', action='store_true')
    parser.add_argument('--use_arabic_nums', action='store_true')
    parser.add_argument('--use_multi_objs', action='store_true')
    parser.add_argument('--use_normalized_semantic_weight', action='store_true')
    parser.add_argument('--use_only_number_word', action='store_true')
    parser.add_argument('--use_random_vector', action='store_true')
    parser.add_argument('--use_ref_with_context', action='store_true')
    parser.add_argument('--use_self_as_ref', action='store_true')
    parser.add_argument('--use_target_obj_with_context', action='store_true')
    parser.add_argument('--not_log_wandb', action='store_true')
    parser.add_argument('--use_trained_linear_vector_as_ref', action='store_true')
    parser.add_argument('--disable_ref_orth', action='store_true')
    parser.add_argument('--disable_target_orth', action='store_true')
    parser.add_argument('--ablation_proj_to_target_aug_embeds', action='store_true')
    parser.add_argument('--ablation_add_to_target_embeds', action='store_true')

    parser.add_argument('--prefix_tunning', action='store_true')
    parser.add_argument("--prefix_length", type=int, default=1)


    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    if args.ref_obj_file is not None and args.use_multi_objs:
        with open(args.ref_obj_file, 'r') as file:
            ref_obj = file.readlines()
        ref_obj = [line.strip() for line in ref_obj]
        ref_save_name = f"{len(ref_obj)}_refs"
    elif args.ref_obj is None:
        ref_obj = None
        ref_save_name = "None"
    else:
        ref_obj = args.ref_obj
        ref_save_name = args.ref_obj
    
    if not os.path.isdir(args.root_folder):
        os.mkdir(args.root_folder)
    # eval_dir = f"{args.model.split('/')[1]}_{args.trained_text_projection_path.replace('.','').replace('/','_')}_{args.dataset}_{args.task}_{ref_save_name}_{args.use_only_number_word}{args.normalize_number_word}_{args.use_random_vector}_{args.use_multi_objs}_{args.use_target_obj_with_context}_{args.not_use_target_aug_sent_with_context}_{args.use_self_as_ref}"
    # if not os.path.isdir(eval_dir):
    #     os.mkdir(eval_dir)

    # run image generation with stable diffucsion
    if args.task == "image_gen" and args.model == "stable_diffusion":
        pretrained_model_name="CompVis/stable-diffusion-v1-4"
        reproduce_stable_diffusion_results(eval_dir,pretrained_model_name,device)
    elif "clip" in args.model or "siglip" in args.model or "blip" in args.model:
        if "clip" in args.model:
            model = CLIPModel.from_pretrained(args.model)
            processor = CLIPProcessor.from_pretrained(args.model)
        elif "siglip" in args.model:
            model = AutoModel.from_pretrained(args.model)
            processor = AutoProcessor.from_pretrained(args.model)
        elif "blip" in args.model:
            model = BlipModel.from_pretrained(args.model)
            processor = AutoProcessor.from_pretrained(args.model)


        number_shift_vectors=None
        prefix_embedding=None
        # if args.load_trained_text_projection:
        #     print(f"Loading trained text projection from {args.trained_text_projection_path}")
        #     model.text_projection.load_state_dict(torch.load(args.trained_text_projection_path))
        if args.load_trained_clip:
            model_dict = torch.load(os.path.join(args.trained_clip_path,"best_model.pth"))
            if "number_shift_vectors" in model_dict.keys():
                number_shift_vectors = model_dict["number_shift_vectors"]
                del model_dict["number_shift_vectors"]
            if args.prefix_tunning and "prefix_embedding" in model_dict.keys():
                prefix_embedding = model_dict["prefix_embedding"]
                del model_dict["prefix_embedding"]
            for key in list(model_dict.keys()):
                if "clip_text_model." in key:
                    model_dict[key.replace("clip_text_model.","")] = model_dict[key]
                    del model_dict[key]
                if "clip_text_projection." in key:
                    model.text_projection.weight.data = model_dict[key]
                    del model_dict[key]
            model.text_model.load_state_dict(model_dict)
            print(f"loading from {args.trained_clip_path}")
        
        for name,param in model.named_parameters():
            param.requires_grad = False
        model = model.to(device)

        if args.dataset=="custom":
            augmented_data = data_augmentation(torch.load(args.custom_data_path))
            
            if args.task == "image_retrievel":
                image_retrievel(
                    model_name=args.model,
                    ref_obj=args.ref_obj,
                    target_data=augmented_data,
                    device=device
                )
            elif args.task == "classification":

                all_exp_results = {
                    "target_obj":[],
                    "acc":[],
                    "sample_size":[]
                }
                for target_obj in augmented_data.keys():

                    exp_results = img_clf(
                        target_data=augmented_data[target_obj],
                        model=model,
                        processor=processor,
                        args=args,
                        target_obj=target_obj,
                        number_shift_vectors=number_shift_vectors,
                        prefix_embedding=prefix_embedding,
                        prefix_length=args.prefix_length,
                        ref_obj=ref_obj,
                        device=device,
                    )
                    if exp_results is None:
                        exit()
                    all_exp_results["target_obj"].append(target_obj)
                    all_exp_results["acc"].append(exp_results["acc"])
                    all_exp_results["sample_size"].append(len(exp_results["gt_labels"]))
                 
                all_exp_results["target_obj"].append("average")
                all_exp_results["acc"].append(np.sum(np.array(all_exp_results["acc"])*np.array(all_exp_results["sample_size"])/np.sum(all_exp_results["sample_size"])))
                
                df = pd.DataFrame([all_exp_results["acc"]],columns=all_exp_results["target_obj"])
                df = df[["average","dogs","cats","lions","chairs","goats","cows","cherries","roses","boats"]]
                
                configs = args.__dict__
                configs["target_obj"] = "average"
                save_path = os.path.join(args.root_folder,generate_file_path(configs,ext="csv"))
                df.to_csv(save_path)

                if not args.not_log_wandb:
                    if args.load_trained_clip:
                        with open(os.path.join(args.trained_clip_path,"metadata.json"), 'r') as file:
                            train_configs = json.load(file)
                        for key in train_configs:
                            configs[f"train_{key}"] = train_configs[key]

                    wandb.init(project="clip_count_new", config=configs, entity="ruisu")
                    wandb.log({f"acc":all_exp_results["acc"][-1]})
                    wandb.log({f"acc_per_obj":all_exp_results["acc"][:-1]})
                    wandb.finish()
                
                    
        elif (args.dataset=="countbench") and (args.task == "classification"):
            img_clf(
                    target_data=torch.load(args.processed_countbench_data_path,map_location=device),
                    model=model,
                    processor=processor,
                    args=args,
                    number_shift_vectors=number_shift_vectors,
                    prefix_embedding=prefix_embedding,
                    prefix_length=args.prefix_length,
                    ref_obj=ref_obj,
                    device=device,
            )
        
        elif ("coco-count" in args.dataset) and (args.task == "classification"):
            img_clf(
                    target_data=torch.load(args.coco_eval_data_path,map_location=device),
                    model=model,
                    processor=processor,
                    args=args,
                    number_shift_vectors=number_shift_vectors,
                    prefix_embedding=prefix_embedding,
                    prefix_length=args.prefix_length,
                    ref_obj=ref_obj,
                    device=device,
            )
        
        

        

    
            
            
            


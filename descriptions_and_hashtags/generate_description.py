import torch

import argparse
import os
import io
import requests
import pandas as pd
from tqdm import tqdm

from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.processors.video_processor import ToTHWC,ToUint8,load_video
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2

import gradio as gr


def batched_sample_embed(videos, model, vis_processor, device, sample_images=8):
    videos_samples = []
    msgs = []

    # load videos from files and sample images
    for v in videos:
        if v.startswith("http") or v.startswith("www"):
            response = requests.get(v)
            d = io.BytesIO(response.content)
        else:
            d = v
        
        video_samples, msg = load_video(
            video_path=d,
            n_frms=sample_images,
            height=224,
            width=224,
            sampling ="uniform", return_msg = True
        )
    
        videos_samples.append(video_samples)
        msgs.append(msg)


    # get context embeddings

    batch = torch.hstack(videos_samples)
    
    encoded_batch = vis_processor.transform(batch)
    #split batches by images in one video
    splited_batch = [v for v in torch.split(encoded_batch, split_size_or_sections=sample_images, dim=1)]
    
    # stack again for encoder
    batch_for_encoding = torch.vstack([v.unsqueeze(0).to(device) for v in splited_batch])
    
    image_emb, _ = model.encode_videoQformer_visual(batch_for_encoding)
    
    # get list of context embeddings
    splited_embeds = torch.split(image_emb, 1)

    return splited_embeds, msgs

def generate_one_text_with_context(model, vis_processor, splited_embed, msg, prompt, device, model_type='llama', num_beams = 1, temperature = 1.0):
    if model_type == 'vicuna':
        chat_state = default_conversation.copy()
    else:
        chat_state = conv_llava_llama_2.copy()
    
    chat = Chat(model, vis_processor, device=device)
    
    chat_state.append_message(chat_state.roles[0], "<Video><ImageHere></Video> "+ msg)
    chat_state.append_message(chat_state.roles[0], prompt)
    
    # chat.ask(prompt, conv=chat_state)
    ans_out = chat.answer(conv=chat_state,
                              img_list=splited_embed,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=100,
                              max_length=1000)
    llm_message = ans_out[0]
    return llm_message

def generate_text_for_batch(model, vis_processor, splited_embeds, msgs, prompt, device, model_type='llama', num_beams = 1, temperature = 1.0):
    llm_outs = []
    for sample_id in range(len(msgs)):
        llm_out = generate_one_text_with_context(
            model, vis_processor,
            [splited_embeds[sample_id]], 
            msgs[sample_id], prompt=prompt, 
            model_type=model_type, 
            num_beams=num_beams, 
            temperature=temperature,
            device=device
        )
        llm_outs.append(llm_out)
        
    return llm_outs

def parse_args():
    parser = argparse.ArgumentParser(
                    prog='Video describer with llama')

    # prompt = "Generate description of video"
    # prompt = "Shortly describe content of video"
    # prompt = "Generate hashtags for video with _ delimeter"
    # prompt = "Generate search for video"
    # prompt = "How to find this video in search"
    # prompt = "What search terms use to find this video"
    parser.add_argument("--prompt", required=False, default="Shortly describe content of video")

    parser.add_argument("--batch_size", required=False, default=20)

    parser.add_argument("--source_csv", required=False, default="yappy_hackaton_2024_400k.csv")
    parser.add_argument("--rank", required=False, default=0)
    parser.add_argument("--stride", required=False, default=1)
    parser.add_argument("--csv_save_prefix", required=False, default="marked_text_last_")
    
    parser.add_argument("--cfg-path", required=False, default="eval_configs/video_llama_eval_withaudio.yaml", help="path to configuration file.") # "eval_configs/video_llama_eval_withaudio.yaml", "eval_configs/video_llama_eval_only_vl.yaml"
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='llama', help="The type of LLM") # vicuna

    parser.add_argument("--options", required=False, default=[])

    args = parser.parse_args()
    return args

def main():
    # Тут инициализация модельки
    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)

    device=f'cuda:{args.gpu_id}'
    
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device=device)
    print('Initialization Finished')

    data = pd.read_csv(args.source_csv)

    batch_size = args.batch_size
    pos = 0
    
    # use if run several process with this
    my_rank = args.rank
    stride = args.stride
    
    prompt = args.prompt
    marked_df = pd.DataFrame(columns=list(data.columns)+["gen_description"])
    
    # while pos<20: # data.shape[0]:
    for pos in tqdm(range(0, data.shape[0], batch_size*stride)):
        slice_iloc = data.iloc[pos:pos+batch_size]
        links = slice_iloc['link'].tolist()
        splited_embeds, msgs = batched_sample_embed(links, model, vis_processor, sample_images=8, device=device)
        outs = generate_text_for_batch(model, vis_processor, splited_embeds, msgs, prompt=prompt, device=device)
    
        new_batch_df = pd.concat([
            slice_iloc["link"].reset_index(drop=True), 
            slice_iloc["description"].reset_index(drop=True), 
            pd.Series(outs)
        ], axis=1, ignore_index=True)
        new_batch_df.columns = marked_df.columns
        marked_df = pd.concat([marked_df, new_batch_df], ignore_index=True)
    
        marked_df.to_csv(f"{args.csv_save_prefix}{my_rank}.csv")
    

if __name__=="__main__":
    main()

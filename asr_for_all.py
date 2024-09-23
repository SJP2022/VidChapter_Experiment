import argparse
import random
import time
import torch
import os
import pickle
from args import get_args_parser, MODEL_DIR
import whisper
import whisperx
import threading
from concurrent.futures import ThreadPoolExecutor
import json

video_root = "/share/common/VideoDatasets/ActivityNet/videos"
asr_pkl_root = "/share_io03_ssd/test2/shijiapeng/ActivityNet_vidchapters_24_9/asr_pkl"
asr_root = "/share/test/shijiapeng/ActivityNet_Whisper_Large_Latest"

def train_on_gpu(gpu_id, vid):

    #判断文件是否存在
    file_path = os.path.join(asr_pkl_root, vid+".pkl")
    if os.path.exists(file_path):
        print("%s —— %s exists." % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), vid)) 
        return
    
    asr_path = os.path.join(asr_root, vid+".json")
    if not os.path.exists(asr_path):
        print("%s asr don't exists. Can't align it!!!" % (vid)) 
        return
    else:
        with open(asr_path, 'r', encoding='utf-8') as f:
            try:
                asr = json.load(f)
            except json.decoder.JSONDecodeError:
                print("%s json.decoder.JSONDecodeError" % vid)
                return

    #线程开始
    print("%s —— %s started with gpu %d." % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), vid, gpu_id)) 

    video = os.path.join(video_root, vid+".mp4") #视频文件路径

    if asr['language']!='en':
        try:
            align_model, metadata = whisperx.load_align_model(language_code=asr['language'], device=torch.device(f'cuda:{gpu_id}'), model_dir=MODEL_DIR)
            print("extract audio")
            audio = whisperx.load_audio(video)
            print("align ASR")
            aligned_asr = whisperx.align(asr["segments"], align_model, metadata, audio, torch.device(f'cuda:{gpu_id}'), return_char_alignments=False)
        except Exception:
            #print("extract audio")
            #audio = whisperx.load_audio(video)
            #print("align ASR")
            #aligned_asr = whisperx.align(asr["segments"], align_model_en[gpu_id], metadata_en[gpu_id], audio, torch.device(f'cuda:{gpu_id}'), return_char_alignments=False)
            print("%s asr language is %s, can't align!!!" % (vid, asr['language'])) 
            return
    else:
        print("extract audio")
        audio = whisperx.load_audio(video)
        print("align ASR")
        aligned_asr = whisperx.align(asr["segments"], align_model_en[gpu_id], metadata_en[gpu_id], audio, torch.device(f'cuda:{gpu_id}'), return_char_alignments=False)

    pickle.dump(aligned_asr, open(file_path, 'wb'))

    #线程结束
    print("%s —— %s finished with gpu %d." % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), vid, gpu_id)) 

if __name__=='__main__':

    # 定义要使用的GPU数量
    num_gpus = 2
    
    # 创建线程列表
    threads = []

    #每个gpu上部署一个whisper model
    align_model_en = []
    metadata_en = []
    for i in range(num_gpus):
        align_item, metadata_item = whisperx.load_align_model(language_code="en", device=torch.device(f'cuda:{i}'), model_dir=MODEL_DIR)
        align_model_en.append(align_item)
        metadata_en.append(metadata_item)
    print('%d whisperX for en loaded.' % (len(align_model_en)))

    #为每个gpu创建一个线程池
    pool = [ThreadPoolExecutor(max_workers=1) for i in range(num_gpus)]
    print('%d pool built.' % (len(pool)))

    videos = os.listdir(video_root)
    for i in range(len(videos)):
        if videos[i].endswith(".mp4"):
            videos[i] = videos[i][:-4]
    random.shuffle(videos)

    #对每个视频创建处理线程
    for i, vid in enumerate(videos):
        gpu_id = i%num_gpus
        t = pool[gpu_id].submit(train_on_gpu, gpu_id, vid)
        threads.append(t)
            
    # 等待所有线程完成
    flag = True
    while flag:
        flag = False
        for t in threads:
            if not t.done():
                flag = True
    print('All subprocesses done.')
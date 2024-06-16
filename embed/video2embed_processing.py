import asyncio
import io
import os
import pickle
import time
from multiprocessing import Pool, Value
from typing import List

import aiohttp
import pandas as pd
import torch
import video_clip

QUEUE_SIZE = 10
NUM_PROCESS = 9
DUMP_ITERS = 1000
VERBOSE = 20


EMBEDDINGS_SAVE_DIR = "./embeddings/"
DATABASE_DIR = "../src/yappy_hackaton_2024_400k.csv"


clips = pd.read_csv(DATABASE_DIR)

n = len(clips) // NUM_PROCESS
chunks = [clips[i : i + n] for i in range(0, len(clips) + 1, n)]

assert sum([len(x) for x in chunks]) == len(clips)


start_time = time.time()
counter = Value("i", 0)


async def async_download(queue: asyncio.Queue, urls: List[str]):
    async with aiohttp.ClientSession() as session:
        for _, url in enumerate(urls):
            async with session.get(url) as response:
                data = await response.read()
            await queue.put((url, data))

        await queue.put(None)
        print("FINISH DOWNLOAD")


def dump_to_file(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "ab") as f:
        pickle.dump(obj, f)


def video2embedding(video: io.BytesIO, model, vls_processor, device, reduction="mean"):
    embedding = video_clip.get_all_video_embeddings(
        [video], model, vls_processor, device
    )
    if reduction == "mean":
        embedding = embedding[0].mean(dim=1).ravel()
    return embedding


async def infer_net(
    queue: asyncio.Queue,
    process_idx: int,
    verbose: int = VERBOSE,
    dump_iters: int = DUMP_ITERS,
):
    device = f"cuda:{3 + process_idx%3}"

    eval_config = "eval_configs/video_clip_v0.2.yaml"
    model, vis_processor = video_clip.load_model(eval_config)
    model = model.to(device)
    model = model.eval()

    res = {}
    with torch.no_grad():
        idx = 0
        while True:
            data = await queue.get()
            if data is None:
                break

            url, video = data
            embedding = video2embedding(io.BytesIO(video), model, vis_processor, device)

            res[url] = embedding.cpu().numpy()

            if idx != 0 and idx % verbose == 0:
                with counter.get_lock():
                    counter.value += verbose

                if process_idx == 0:
                    print(
                        f"num processed videos: {counter.value}, elapsed time {(time.time() - start_time)} s"
                    )

            if idx != 0 and idx % dump_iters == 0:
                file_path = os.path.join(
                    EMBEDDINGS_SAVE_DIR, f"procces_idx_{process_idx}_iter_{idx}"
                )
                dump_to_file(file_path, res)
                res = {}

            idx += 1

    print("FINISH INFER", process_idx)
    file_path = os.path.join(
        EMBEDDINGS_SAVE_DIR, f"procces_idx_{process_idx}_iter_{idx}"
    )
    dump_to_file(file_path, res)


def run(chunk: pd.DataFrame, process_idx: int):
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue(maxsize=QUEUE_SIZE)

    downloader = loop.create_task(async_download(queue, chunk["link"].tolist()))
    consumer_task = loop.create_task(infer_net(queue, process_idx))

    loop.run_until_complete(asyncio.gather(*[downloader, consumer_task]))
    loop.close()


with Pool(NUM_PROCESS) as p:
    results = []
    for i in range(NUM_PROCESS):
        results.append(p.apply_async(run, [chunks[i], i]))

    for res in results:
        res.get()

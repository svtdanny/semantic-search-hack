# DL Semantic search over video

__Team:__ ML бригада

### YouTube Overview

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/3V3iHjHVNOI/maxresdefault.jpg)](https://www.youtube.com/watch?v=3V3iHjHVNOI)

### Docker 
if you use ssh port-forwarding on remote vm, you need to use `--net=host` with docker run

`docker pull sivtsovdt/hkt-base:all-v0` \
// or \
`docker build -t hack-backend .` \
`docker tag hack-backend sivtsovdt/hkt-base:all-v0`

It will take a long time... \
`docker run -it --net=host -p 8000:8000 sivtsovdt/hkt-base:all-v0`

# Download embeddings

`curl -L https://clck.ru/3BJSJ7 -o ./embed/joined_embeddings.pkl`

# Download model

`git lfs install` \
`git clone https://huggingface.co/AskYoutube/AskVideos-VideoCLIP-v0.2` \
`cd embed` \
`mkdir models` \
`cp ../AskVideos-VideoCLIP-v0.2/askvideos_clip_v0.2.pth ./models`

### Run from source

Add path to root and to every directory you are going to use

`PYTHONPATH=$PYTHONPATH:~/semantic-search-hack:~/semantic-search-hack/embed fastapi dev backend/service.py`

### frontend

You can use same environment for launch frontend, launch example

```bash
cd frontend
 python3 -m streamlit run main.py --server.port 6884 -- --address 127.0.0.1:8000
```

# DL Semantic search over video

__Team:__ ML бригада

### Docker 
if you use ssh port-forwarding on remote vm, you need to use `--net=host` with docker run

`docker pull sivtsovdt/hkt-base:all-v0` \
// or \
`docker build -t hack-backend .` \
`docker tag hack-backend sivtsovdt/hkt-base:all-v0`

`docker run -it --net=host -p 8000:8000 sivtsovdt/hkt-base:all-v0`

# Download embeddings

`curl -L https://clck.ru/3BJSJ7 -o ./embed/joined_embeddings.pkl`

# Download model

`git lfs install` \
`git clone https://huggingface.co/AskYoutube/AskVideos-VideoCLIP-v0.2` \
`cd embed` \
`mkdir models` \
`cp ../AskVideos-VideoCLIP-v0.2/askvideos_clip_v0.2.pth ./models` \

### Run from source

Add path to root and to every directory you are going to use

`PYTHONPATH=$PYTHONPATH:~/semantic-search-hack:~/semantic-search-hack/embed fastapi dev backend/service.py`

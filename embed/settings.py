class Settings:
    device: str = "cpu"
    eval_config: str = "embed/eval_configs/video_clip_v0.2.yaml"
    precalculated_embeddings: str = "embed/joined_embeddings.pkl"
    use_qdrant_index: bool = True


settings = Settings()

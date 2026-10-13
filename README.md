# ncf-movielens

Movie rating prediction on MovieLens 100K using neural collaborative filtering in PyTorch.

This is a rebuild of my [CS final project](https://github.com/jonahantony147/cs-sem1-finalproject), which used k-nearest neighbors. Here each user and each movie gets a learned 32-dim embedding, and a small MLP predicts the rating from the two embeddings. I also wanted to see how much faster training is on my GPU, so the script trains the same model on CPU and then on GPU and times both.

## Results

80/20 train/test split, 20 epochs, batch size 1024, Adam (lr 1e-3).

| Device | Test RMSE | Train time |
|---|---|---|
| CPU | 0.9664 | 5.1 s |
| RTX 4060 Laptop GPU | 0.9686 | 3.3 s |

Only about 1.5x faster on the GPU. The model and batches are small, so the GPU doesn't have much work per step. I expect the gap to get bigger with larger embeddings or batch sizes, which is what I want to try next.

## Running it

```
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install pandas numpy requests
python ncf_movielens.py
```

The first line installs the CUDA version of PyTorch (a plain `pip install torch` on Windows is CPU only). Without a GPU it'll still run, just the CPU part.

It looks for an `ml-100k` folder or `ml-100k.zip` next to the script and downloads the dataset if neither is there.

## Data

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) from GroupLens Research.

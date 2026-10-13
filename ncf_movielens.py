# neural collaborative filtering on movielens 100k
# trains the same model on cpu and gpu so i can compare the times

import io, os, time, zipfile
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn

# load the data, uses a local ml-100k folder or zip if it's there, otherwise downloads it
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["user", "item", "rating", "ts"]
local_data = os.path.join(HERE, "ml-100k", "u.data")
local_zip = os.path.join(HERE, "ml-100k.zip")
if os.path.exists(local_data):
    df = pd.read_csv(local_data, sep="\t", names=COLS)
else:
    src = local_zip if os.path.exists(local_zip) else io.BytesIO(requests.get(URL, timeout=60).content)
    with zipfile.ZipFile(src) as z:
        df = pd.read_csv(z.open("ml-100k/u.data"), sep="\t", names=COLS)

# ids start at 1, embeddings need them to start at 0
df["user"] -= 1
df["item"] -= 1
n_users, n_items = df["user"].max() + 1, df["item"].max() + 1

# 80/20 split
rng = np.random.default_rng(42)
mask = rng.random(len(df)) < 0.8
train, test = df[mask], df[~mask]

def to_tensors(d):
    return (torch.tensor(d["user"].values), torch.tensor(d["item"].values),
            torch.tensor(d["rating"].values, dtype=torch.float32))

# each user and movie gets its own vector, the mlp predicts the rating from both
class NCF(nn.Module):
    def __init__(self, n_users, n_items, dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, u, i):
        x = torch.cat([self.user_emb(u), self.item_emb(i)], dim=1)
        return self.mlp(x).squeeze(1)

def run(device, epochs=20, batch_size=1024):
    torch.manual_seed(0)
    model = NCF(n_users, n_items).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    u_tr, i_tr, r_tr = [t.to(device) for t in to_tensors(train)]
    u_te, i_te, r_te = [t.to(device) for t in to_tensors(test)]

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(r_tr), device=device)
        for b in range(0, len(perm), batch_size):
            idx = perm[b:b + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(u_tr[idx], i_tr[idx]), r_tr[idx])
            loss.backward()
            opt.step()
    # gpu runs async so wait for it to finish before stopping the timer
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    model.eval()
    with torch.no_grad():
        preds = model(u_te, i_te).clamp(1, 5)  # ratings only go 1 to 5
        rmse = torch.sqrt(((preds - r_te) ** 2).mean()).item()
    return rmse, elapsed

cpu_rmse, cpu_time = run("cpu")
print(f"CPU: test RMSE {cpu_rmse:.4f}, training time {cpu_time:.1f}s")

if torch.cuda.is_available():
    gpu_rmse, gpu_time = run("cuda")
    print(f"GPU ({torch.cuda.get_device_name(0)}): test RMSE {gpu_rmse:.4f}, "
          f"training time {gpu_time:.1f}s")
    print(f"GPU speedup: {cpu_time / gpu_time:.1f}x")
else:
    print("no gpu found, check that torch was installed with cuda")

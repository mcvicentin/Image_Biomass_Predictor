import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.models.model import ImageOnlyMultitaskNet

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    base = Path(".")

    ckpt_path = base / "models" / "weights" / "image_only_ckpt.pt"
    test_csv  = base / "data" / "raw" / "test.csv"
    emb_csv   = base / "data" / "external" / "embeddings" / "image_embeddings.csv"
    out_csv   = base / "outputs" / "submission.csv"

    # Carregar checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    img_dim = ckpt["img_dim"]
    n_aux_num = ckpt["n_aux_num"]
    n_classes = ckpt["n_classes"]
    target_order = ckpt["target_order"]

    model = ImageOnlyMultitaskNet(
        img_dim=img_dim,
        n_aux_num=n_aux_num,
        n_classes_dict=n_classes,
        hidden=256,
        dropout=0.2,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # Test dataset
    test = pd.read_csv(test_csv)
    test["image_id"] = test["image_path"].str.extract(r"(ID\d+)")

    # Embeddings
    emb = pd.read_csv(emb_csv)
    if "image_id" not in emb:
        emb["image_id"] = emb["image_path"].str.extract(r"(ID\d+)")
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]

    test = test.merge(
        emb[["image_id"] + emb_cols],
        on="image_id",
        how="left"
    )

    missing = test[emb_cols].isna().sum().sum()
    print("Missing embeddings:", missing)
    if missing > 0:
        test = test.dropna(subset=emb_cols).reset_index(drop=True)

    # Predição por imagem única
    unique = (
        test[["image_id"] + emb_cols]
        .drop_duplicates("image_id")
        .reset_index(drop=True)
    )

    X = torch.tensor(unique[emb_cols].values, dtype=torch.float32).to(device)

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out = model(X[i:i+256])
            preds.append(out["biom"].cpu().numpy())

    preds = np.vstack(preds)
    id_to_pred = dict(zip(unique["image_id"], preds))

    name_to_idx = {name: i for i, name in enumerate(target_order)}

    def pred_row(row):
        vec = id_to_pred[row["image_id"]]
        idx = name_to_idx[row["target_name"]]
        return float(vec[idx])

    test["target"] = test.apply(pred_row, axis=1)

    submission = test[["sample_id", "target"]]
    out_csv.parent.mkdir(exist_ok=True, parents=True)
    submission.to_csv(out_csv, index=False)

    print("Salvo:", out_csv)


if __name__ == "__main__":
    main()

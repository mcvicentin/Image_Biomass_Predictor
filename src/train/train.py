import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm

from src.data.dataset import PastureImageOnlyDataset
from src.models.model import ImageOnlyMultitaskNet
from src.utils.misc import rmse_np, weighted_r2_score

# -------- CONFIGURAÇÕES ORIGINAIS DO NOTEBOOK --------
target_order = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]
aux_num_cols = ["Pre_GSHH_NDVI", "Height_Ave_cm"]
aux_cat_cols = ["month", "Season", "State", "Species"]

weights_dict = {
    "Dry_Clover_g": 0.1,
    "Dry_Dead_g":   0.1,
    "Dry_Green_g":  0.1,
    "Dry_Total_g":  0.5,
    "GDM_g":        0.2,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------- FUNÇÃO compute_loss ORIGINAL --------
def compute_loss(outputs, y_biom, y_aux_num, y_aux_cat, alpha_aux=0.0):
    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()

    loss_biom = mse(outputs["biom"], y_biom)

    if alpha_aux <= 0.0:
        return loss_biom, {"loss_biom": loss_biom.item()}

    loss_aux_num = mse(outputs["aux_num"], y_aux_num)

    loss_month   = ce(outputs["month_logits"],  y_aux_cat["month"])
    loss_season  = ce(outputs["season_logits"], y_aux_cat["Season"])
    loss_state   = ce(outputs["state_logits"],  y_aux_cat["State"])
    loss_species = ce(outputs["species_logits"],y_aux_cat["Species"])

    loss_aux_cat = (loss_month + loss_season + loss_state + loss_species) / 4.0

    loss_total = loss_biom + alpha_aux * (loss_aux_num + loss_aux_cat)

    return loss_total, {
        "loss_biom": loss_biom.item(),
        "loss_aux_num": loss_aux_num.item(),
        "loss_aux_cat": loss_aux_cat.item(),
        "loss_total": loss_total.item(),
    }


# -------- FUNÇÃO train_one_run ORIGINAL --------
def train_one_run(
    wide_train,
    *,
    emb_cols,
    target_order,
    aux_num_cols,
    aux_cat_cols,
    n_classes,
    alpha_aux=0.0,
    val_frac=0.2,
    epochs=50,
    patience=8,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    verbose=True,
):
    n = len(wide_train)
    idx_all = np.arange(n)
    idx_tr, idx_val = train_test_split(
        idx_all, test_size=val_frac, random_state=42
    )

    wide_tr  = wide_train.iloc[idx_tr].reset_index(drop=True)
    wide_val = wide_train.iloc[idx_val].reset_index(drop=True)

    ds_tr  = PastureImageOnlyDataset(wide_tr, emb_cols, target_order, aux_num_cols, aux_cat_cols)
    ds_val = PastureImageOnlyDataset(wide_val, emb_cols, target_order, aux_num_cols, aux_cat_cols)

    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    img_dim = len(emb_cols)
    model = ImageOnlyMultitaskNet(
        img_dim=img_dim,
        n_aux_num=len(aux_num_cols),
        n_classes_dict=n_classes,
        hidden=256,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = np.inf
    best_state = None
    wait = 0

    if verbose:
        print("\n=== Training one run (image-only) ===\n")

    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs")

    for epoch in epoch_bar:
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0

        for x_img, y_biom, y_aux_num, y_aux_cat in train_loader:
            x_img     = x_img.to(device)
            y_biom    = y_biom.to(device)
            y_aux_num = y_aux_num.to(device)
            y_aux_cat_dev = {k: v.to(device) for k, v in y_aux_cat.items()}

            optimizer.zero_grad()
            outputs = model(x_img)
            loss, _ = compute_loss(
                outputs, y_biom, y_aux_num, y_aux_cat_dev,
                alpha_aux=alpha_aux
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- VAL ----
        model.eval()
        val_loss = 0.0
        preds_biom, targs_biom = [], []

        with torch.no_grad():
            for x_img, y_biom, y_aux_num, y_aux_cat in val_loader:
                x_img  = x_img.to(device)
                y_biom = y_biom.to(device)

                outputs = model(x_img)
                loss, _ = compute_loss(
                    outputs, y_biom,
                    y_aux_num=None,  
                    y_aux_cat=None,
                    alpha_aux=0.0 
                )

                val_loss += loss.item()
                preds_biom.append(outputs["biom"].cpu().numpy())
                targs_biom.append(y_biom.cpu().numpy())

        val_loss /= len(val_loader)
        preds_biom = np.vstack(preds_biom)
        targs_biom = np.vstack(targs_biom)

        rmse_val = rmse_np(targs_biom, preds_biom)
        r2_val   = r2_score(targs_biom.ravel(), preds_biom.ravel())
        r2w_val  = weighted_r2_score(targs_biom, preds_biom, weights_dict, target_order)

        epoch_bar.set_postfix({
            "train_loss": f"{train_loss:.1f}",
            "val_loss":   f"{val_loss:.1f}",
            "RMSE":       f"{rmse_val:.2f}",
            "R2":         f"{r2_val:.3f}",
            "R2w":        f"{r2w_val:.3f}",
        })

        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print("\nEarly stopping.\n")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# -------- FUNÇÃO run_blind_experiments_image_only ORIGINAL --------
def run_blind_experiments_image_only(
    wide,
    n_runs=5,
    blind_frac=0.2,
    alpha_aux=0.0,
    seed_base=42,
    emb_cols=None,
    n_classes=None,
):
    results = []
    all_true = []
    all_pred = []

    n = len(wide)

    for run in range(n_runs):
        seed = seed_base + run
        rng = np.random.default_rng(seed)
        idx_all = np.arange(n)
        rng.shuffle(idx_all)

        blind_size = int(blind_frac * n)
        blind_idx = idx_all[:blind_size]
        train_idx = idx_all[blind_size:]

        wide_train = wide.iloc[train_idx].reset_index(drop=True)
        wide_blind = wide.iloc[blind_idx].reset_index(drop=True)

        model = train_one_run(
            wide_train,
            emb_cols=emb_cols,
            target_order=target_order,
            aux_num_cols=aux_num_cols,
            aux_cat_cols=aux_cat_cols,
            n_classes=n_classes,
            alpha_aux=alpha_aux,
            epochs=50,
            patience=8,
            verbose=True,
        )

        ds_blind = PastureImageOnlyDataset(
            wide_blind, emb_cols, target_order, aux_num_cols, aux_cat_cols
        )
        loader_blind = DataLoader(ds_blind, batch_size=64, shuffle=False)

        model.eval()
        preds_blind = []
        targs_blind = []

        with torch.no_grad():
            for x_img, y_biom, y_aux_num, y_aux_cat in loader_blind:
                x_img  = x_img.to(device)
                y_biom = y_biom.to(device)

                outputs = model(x_img)
                preds_blind.append(outputs["biom"].cpu().numpy())
                targs_blind.append(y_biom.cpu().numpy())

        preds_blind = np.vstack(preds_blind)
        targs_blind = np.vstack(targs_blind)

        rmse_b = rmse_np(targs_blind, preds_blind)
        r2_b   = r2_score(targs_blind.ravel(), preds_blind.ravel())
        r2w_b  = weighted_r2_score(targs_blind, preds_blind, weights_dict, target_order)

        results.append({
            "run": run,
            "rmse":  rmse_b,
            "r2":    r2_b,
            "r2w":   r2w_b,
        })

        all_true.append(targs_blind)
        all_pred.append(preds_blind)

        print(f"\nRun {run}: RMSE={rmse_b:.2f}, R²={r2_b:.3f}, R²w={r2w_b:.3f}")

    results_df = pd.DataFrame(results)

    all_true = np.vstack(all_true)
    all_pred = np.vstack(all_pred)

    return results_df, all_true, all_pred


# -------- ROTINA PRINCIPAL DO SCRIPT --------
def main():
    base = Path(".")
    train_csv = base / "data" / "raw" / "train.csv"
    emb_csv   = base / "data" / "external" / "embeddings" / "image_embeddings.csv"

    print("Carregando train:", train_csv)
    train = pd.read_csv(train_csv)

    if not np.issubdtype(train["Sampling_Date"].dtype, np.datetime64):
        train["Sampling_Date"] = pd.to_datetime(train["Sampling_Date"])

    train["image_id"] = train["image_path"].str.extract(r"(ID\d+)")

    train["month"] = train["Sampling_Date"].dt.month
    train["Season"] = train["Sampling_Date"].dt.month % 12 // 3 + 1

    pivot_index_cols = [
        "image_id","image_path","Sampling_Date","State","Species",
        "Pre_GSHH_NDVI","Height_Ave_cm","month","Season"
    ]

    wide = (
        train
        .pivot_table(
            index=pivot_index_cols,
            columns="target_name",
            values="target"
        )
        .reset_index()
    )

    wide = wide[pivot_index_cols + target_order]

    emb = pd.read_csv(emb_csv)
    if "image_id" not in emb.columns:
        emb["image_id"] = emb["image_path"].str.extract(r"(ID\d+)")
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]

    wide = wide.merge(
        emb[["image_id"] + emb_cols],
        on="image_id",
        how="left"
    )

    missing_emb = wide[emb_cols].isna().sum().sum()
    print("Missing embeddings:", missing_emb)
    if missing_emb > 0:
        wide = wide.dropna(subset=emb_cols).reset_index(drop=True)

    # codificação
    for col in aux_cat_cols:
        wide[col] = wide[col].astype("category")
        wide[col + "_code"] = wide[col].cat.codes.astype("int64")

    n_classes = {
        col: wide[col + "_code"].max() + 1
        for col in aux_cat_cols
    }

    # === Rodar exatamente como no notebook ===
    results_df, all_true_img, all_pred_img = run_blind_experiments_image_only(
        wide,
        n_runs=10,
        blind_frac=0.2,
        alpha_aux=0.0,
        seed_base=42,
        emb_cols=emb_cols,
        n_classes=n_classes,
    )

    print("Resultados médios:\n", results_df.mean())

    rmse_global = rmse_np(all_true_img, all_pred_img)
    r2_global   = r2_score(all_true_img.ravel(), all_pred_img.ravel())
    r2w_global  = weighted_r2_score(all_true_img, all_pred_img, weights_dict, target_order)

    print("\n== Métricas agregadas ==")
    print(f"RMSE: {rmse_global:.4f}")
    print(f"R²:   {r2_global:.4f}")
    print(f"R²w:  {r2w_global:.4f}")

    # salvar CSV igual ao notebook
    cols_true = [f"true_{t}" for t in target_order]
    cols_pred = [f"pred_{t}" for t in target_order]

    df_pairs = pd.DataFrame(
        np.hstack([all_true_img, all_pred_img]),
        columns=cols_true + cols_pred
    )

    out_csv = base / "outputs" / "blind_true_vs_pred_image_only.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_pairs.to_csv(out_csv, index=False)

    print("\nSalvo:", out_csv)


if __name__ == "__main__":
    main()

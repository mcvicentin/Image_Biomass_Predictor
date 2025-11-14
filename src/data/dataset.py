import torch
from torch.utils.data import Dataset


class PastureImageOnlyDataset(Dataset):
    """
    Dataset exatamente como no notebook.
    """
    def __init__(self, wide_df, emb_cols, target_order, aux_num_cols, aux_cat_cols):
        self.X = torch.tensor(
            wide_df[emb_cols].values.astype("float32"),
            dtype=torch.float32
        )
        self.y_biom = torch.tensor(
            wide_df[target_order].values.astype("float32"),
            dtype=torch.float32
        )
        self.y_aux_num = torch.tensor(
            wide_df[aux_num_cols].values.astype("float32"),
            dtype=torch.float32
        )

        # categorias
        self.y_aux_cat = {
            col: torch.tensor(
                wide_df[col + "_code"].values.astype("int64"),
                dtype=torch.long
            )
            for col in aux_cat_cols
        }

        self.aux_cat_cols = aux_cat_cols

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_img = self.X[idx]
        y_biom = self.y_biom[idx]
        y_aux_num = self.y_aux_num[idx]
        y_aux_cat = {col: self.y_aux_cat[col][idx] for col in self.aux_cat_cols}

        return x_img, y_biom, y_aux_num, y_aux_cat

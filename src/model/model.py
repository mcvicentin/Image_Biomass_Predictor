import torch
import torch.nn as nn


class ImageOnlyMultitaskNet(nn.Module):
    def __init__(
        self,
        img_dim,
        n_aux_num=2,
        n_classes_dict=None,
        hidden=256,
        dropout=0.2,
    ):
        super().__init__()

        if n_classes_dict is None:
            raise ValueError("n_classes_dict deve ser passado.")

        # Backbone
        self.backbone = nn.Sequential(
            nn.Linear(img_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        # Head principal (biomassa)
        self.head_biom = nn.Linear(hidden, 5)

        # Head auxiliar numérica
        self.head_aux_num = nn.Linear(hidden, n_aux_num)

        # Heads categóricas
        self.head_month  = nn.Linear(hidden, n_classes_dict["month"])
        self.head_season = nn.Linear(hidden, n_classes_dict["Season"])
        self.head_state  = nn.Linear(hidden, n_classes_dict["State"])
        self.head_species = nn.Linear(hidden, n_classes_dict["Species"])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_img):
        h = self.backbone(x_img)

        return {
            "biom": self.head_biom(h),
            "aux_num": self.head_aux_num(h),
            "month_logits":  self.head_month(h),
            "season_logits": self.head_season(h),
            "state_logits":  self.head_state(h),
            "species_logits": self.head_species(h),
        }

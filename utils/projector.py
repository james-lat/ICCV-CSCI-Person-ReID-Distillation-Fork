from timm.layers import Mlp
import torch.nn as nn

class PR(nn.Module):
    def __init__(self, backbone, in_dim=768, out_dim=1024, hidden=384, drop=0.0):
        super().__init__()
        self.backbone = backbone
        self.proj = Mlp(
            in_features=in_dim,
            hidden_features=hidden,
            out_features=out_dim,
            drop=drop,
            act_layer=nn.GELU,
        )

    def forward(self, *args, **kwargs):
        # call the EVA backbone normally
        out = self.backbone(*args, **kwargs)

        # normalize backbone outputs into (score, feat_768)
        if isinstance(out, (list, tuple)):
            if len(out) == 3:
                score, feat_768, _ = out           # ignore any third thing from EVA
            elif len(out) >= 2:
                score, feat_768 = out[0], out[1]
            else:
                score, feat_768 = None, out
        else:
            score, feat_768 = None, out

        # project 768-d feature → 1024-d student feature for KD
        student_1024 = self.proj(feat_768)

        # ALWAYS return the KD feature as the 3rd element
        return score, feat_768, student_1024

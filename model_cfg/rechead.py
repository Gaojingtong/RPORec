import torch
from torch import nn
from sentence_transformers import SentenceTransformer

class MulRetriever(nn.Module):
    def __init__(self, model_names="tomaarsen/static-retrieval-mrl-en-v1", embed_dim=1024, num_heads=4, qkv_structure=[128]):
        super().__init__()
        self.retriever1 = SentenceTransformer(model_names)
        self.retriever2 = SentenceTransformer(model_names)
        self.retriever3 = SentenceTransformer(model_names)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=4)


        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, 3*embed_dim),
            nn.Sigmoid()
        )

        self.final_gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        self.prompt_fn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        self.cot_fn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim
        self.layernorm = nn.LayerNorm(embed_dim)


    def forward(self, x, y, z, format_acc, item_h):
        # with torch.no_grad():
        #     format_acc[:1]=format_acc[:1]+1.0
        # Encode all inputs in parallel
        re_x = self.retriever1(x)['sentence_embedding']
        re_y = self.retriever2(y)['sentence_embedding']
        re_z = self.retriever3(z)['sentence_embedding']
        re_x = self.prompt_fn(re_x)
        re_y = self.cot_fn(re_y)

        # 3, b, h
        re = torch.cat((re_x.unsqueeze(0), re_y.unsqueeze(0), re_z.unsqueeze(0)),
                       dim=0)
        # b,3,h
        re = re.permute(1, 0, 2)
        # b,h
        re_select = torch.bmm(format_acc.unsqueeze(1), re).squeeze()

        re = torch.cat((re_x.unsqueeze(0), re_y.unsqueeze(0), re_select.unsqueeze(0)),
                       dim=0)
        # re = re.permute(1, 0, 2)

        # with torch.no_grad():
        #     re_dist = torch.sum(re_select**2, dim=1)**0.5
        #     re_dist = re_dist.unsqueeze(1)
        re = self.trans(re)

        attn_out = re[-1, :, :].squeeze()
        # attn_out = attn_out/(torch.sum(attn_out**2, dim=1)**0.5).unsqueeze(1)
        # re_select = re_select/(torch.sum(re_select**2, dim=1)**0.5).unsqueeze(1)
        # b,2,h
        re_out = torch.cat((re_select.unsqueeze(1), attn_out.unsqueeze(1)),
                            dim=1)
        # b,1,2
        final_gate = self.final_gate(re_out.reshape((-1, 2 * self.embed_dim)))*0.1
        re = final_gate*attn_out+re_select
        re = re.squeeze()

        # re = re / (torch.sum(re**2, dim=1)**0.5).unsqueeze(-1)
        if item_h.dim() == 2:
            # with torch.no_grad():
                # item_h = item_h / (torch.sum(item_h**2, dim=-1)**0.5).unsqueeze(-1)
            # Original case: (num_items, embed_dim)
            re = re @ item_h.t()
        else:
            # with torch.no_grad():
                # item_h = item_h / (torch.sum(item_h**2, dim=-1)**0.5).unsqueeze(-1)
            # Negative sampling case: (batch_size, 1+k, embed_dim)
            re = torch.bmm(re.unsqueeze(1), item_h.transpose(1, 2)).squeeze(1)
        return re
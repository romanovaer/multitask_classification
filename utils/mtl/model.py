from torch import nn


class Doc2Vec(nn.Module):
    def __init__(self, num_embeddings, dim_embeddings=100):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, dim_embeddings)
        self.cls_1 = nn.Linear(in_features=dim_embeddings, out_features=1)
        self.cls_2 = nn.Linear(in_features=dim_embeddings, out_features=1)

    def forward(self, x):
        x = self.embed(x)
        x = x.mean(dim=1).squeeze(1)
        return self.cls_1(x).squeeze(-1), self.cls_2(x).squeeze(-1)


class BiTaskLSTMModel(nn.Module):
    def __init__(self, num_embeddings, dim_embeddings=100):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, dim_embeddings)
        self.lstm = nn.LSTM(input_size=dim_embeddings,
                            hidden_size=dim_embeddings,
                            batch_first=True)
        self.cls_1 = nn.Linear(in_features=dim_embeddings, out_features=1)
        self.cls_2 = nn.Linear(in_features=dim_embeddings, out_features=1)

    def forward(self, x):
        x = self.embed(x)
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        return self.cls_1(output).squeeze(-1), self.cls_2(output).squeeze(-1)


class MultiTaskLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, fact):

        loss_func_0 = nn.BCEWithLogitsLoss()
        loss_func_1 = nn.BCEWithLogitsLoss()

        loss_0 = loss_func_0(preds[0], fact[:, 0].float())
        loss_1 = loss_func_1(preds[1], fact[:, 1].float())

        return (loss_0 + loss_1) / 2

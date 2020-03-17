import string
import random
import re
import math
import csv

import numpy as np
import torch
import torchtext
from torchtext.vocab import Vectors
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def preprocessing_text(text):
    # 改行コードを消去
    text = re.sub('<br />', '', text)

    # カンマ、ピリオド以外の記号をスペースに置換
    # # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")

    # ピリオドなどの前後にはスペースを入れておく
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    return text


def tokenizer_punctuation(text):
    return text.strip().split()


# 前処理と分かち書きをまとめた関数を定義
def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    ret = tokenizer_punctuation(text)
    return ret


###########
# data loader
###########
max_length = 256
TEXT = torchtext.data.Field(
    sequential=True,  # 可変長のtext
    tokenize=tokenizer_with_preprocessing,  # 前処理&tokenizeする関数
    use_vocab=True,
    lower=True,  # 小文字に変換
    include_lengths=True,  # 長さの保存.padding済みの行列とtextの長さを表す配列をreturn
    batch_first=True,  # バッチの次元を[0]
    fix_length=max_length,  # 最大長
    init_token="<cls>",  # 文頭を埋める
    eos_token="<eos>")
LABEL = torchtext.data.Field(sequential=False, use_vocab=False)


# フォルダ「data」から各tsvファイルを読み込みます
print('reading_data')
train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path='./data/', train='IMDb_train.tsv',
    test='IMDb_test.tsv', format='tsv',
    fields=[('Text', TEXT), ('Label', LABEL)])

# trainとvalに分割
train_ds, val_ds = train_val_ds.split(split_ratio=0.8, random_state=random.seed(1234))

# voabの作成
print('reading_vec')
english_fasttext_vectors = Vectors(name='data/wiki-news-300d-1M.vec')
TEXT.build_vocab(train_ds, vectors=english_fasttext_vectors, min_freq=10)
print('shape_of_vocab:', TEXT.vocab.vectors.shape)

# Dataloader
train_dl = torchtext.data.Iterator(train_ds, batch_size=24, train=True)
val_dl = torchtext.data.Iterator(val_ds, batch_size=24, train=False, sort=False)
test_dl = torchtext.data.Iterator(test_ds, batch_size=24, train=False, sort=False)



###########
# NN
###########

class Embedder(nn.Module):
    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors,
            freeze=True
        )

    def forward(self, x):
        x_vec = self.embeddings(x)

        return x_vec


class PositionalEncoder(nn.Module):
    def __init__(self, d_model=300, max_seq_len=256):
        super().__init__()

        self.d_model = d_model  # 単語ベクトルの次元数

        # 単語の順番（pos）と埋め込みベクトルの次元の位置（i）によって一意に定まる値の表をpeとして作成
        pe = torch.zeros(max_seq_len, d_model)

        # GPUが使える場合はGPUへ送る、ここでは省略。実際に学習時には使用する
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * (i + 1))/d_model)))

        # 表peの先頭に、ミニバッチ次元となる次元を足す
        self.pe = pe.unsqueeze(0)

        # 勾配を計算しないようにする
        self.pe.requires_grad = False

    def forward(self, x):

        # 入力xとPositonal Encodingを足し算する
        # xがpeよりも小さいので、大きくする
        ret = math.sqrt(self.d_model)*x + self.pe
        return ret


class Attention(nn.Module):
    def __init__(self, d_model=300):
        super().__init__()

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out=nn.Linear(d_model, d_model)

        self.d_k = d_model

    def forward(self, q, k, v, mask):
        q = self.q_linear(q)
        k = self.q_linear(k)
        v = self.q_linear(v)

        weights = torch.matmul(q, k.transpose(1, 2) / math.sqrt(self.d_k))

        # maskは小さい値に置換
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)

        # softmaxで規格化をする
        normlized_weights = F.softmax(weights, dim=-1)

        # AttentionをValueとかけ算
        output = torch.matmul(normlized_weights, v)

        # 全結合層で特徴量を変換
        output = self.out(output)
        return output, normlized_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        # LayerNormalization層
        # https://pytorch.org/docs/stable/nn.html?highlight=layernorm
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        # Attention層
        self.attn = Attention(d_model)

        # Attentionのあとの全結合層2つ
        self.ff = FeedForward(d_model)

        # Dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 正規化とAttention
        x_normlized = self.norm_1(x)
        output, normlized_weights = self.attn(x_normlized, x_normlized, x_normlized, mask)
        x2 = x + self.dropout_1(output)
        # 正規化と全結合層
        x_normlized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normlized2))
        return output, normlized_weights


class ClassificationHead(nn.Module):
    def __init__(self, d_model=300, output_dim=2):
        super().__init__()

        # 全結合層
        self.linear = nn.Linear(d_model, output_dim)  # output_dimはポジ・ネガの2つ

        # 重み初期化処理
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, x):
        x0 = x[:, 0, :]  # バッチの文頭<cls>の特徴量
        out = self.linear(x0)

        return out


class TransformerClassification(nn.Module):
    def __init__(self, text_embedding_vectors, d_model=300, max_seq_len=256, output_dim=2):
        super().__init__()

        # モデル構築
        self.net1 = Embedder(text_embedding_vectors)
        self.net2 = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.net3_1 = TransformerBlock(d_model=d_model)
        self.net3_2 = TransformerBlock(d_model=d_model)
        self.net4 = ClassificationHead(output_dim=output_dim, d_model=d_model)

    def forward(self, x, mask):
        x1 = self.net1(x)  # 単語をベクトルに
        x2 = self.net2(x1)  # Positon情報を足し算
        x3_1, normlized_weights_1 = self.net3_1(x2, mask)  # Self-Attentionで特徴量を変換
        x3_2, normlized_weights_2 = self.net3_2(x3_1, mask)  # Self-Attentionで特徴量を変換
        x4 = self.net4(x3_2)  # 最終出力の0単語目を使用して、分類0-1のスカラーを出力
        return x4, normlized_weights_1, normlized_weights_2


############
# 学習
###########
# 可視化
def plot_history(csv_path, png_path):
    df = pd.read_csv(csv_path)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df['epoch'], df['train_loss'], label='g_train')
    ax.plot(df['epoch'], df['val_loss'], label='d_train')
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    fig.savefig(png_path)


# dataloaderをまとめる
dataloaders_dict = {"train": train_dl, "val": val_dl}

# nnの構築
net = TransformerClassification(text_embedding_vectors=TEXT.vocab.vectors, d_model=300, max_seq_len=256, output_dim=2)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Liner層の初期化
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
net.train()
net.net3_1.apply(weights_init)
net.net3_2.apply(weights_init)


# 損失関数の設定
criterion = nn.CrossEntropyLoss()
# 最適化手法の設定
learning_rate = 2e-5
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# GPU関連
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)
print('-----start-------')
# ネットワークをGPUへ
net.to(device)
# ネットワークがある程度固定であれば、高速化させる
torch.backends.cudnn.benchmark = True

# epochのループ
train_loss_list = []
val_loss_list = []

csv_path = './history.csv'
png_path = './history.png'
with open(csv_path, 'w') as f:
    writer = csv.writer(f)
    header = ['epoch', 'train_loss', 'val_loss']
    writer.writerow(header)
num_epochs = 10
for epoch in range(num_epochs):
# epochごとの訓練と検証のループ
    for phase in ['train', 'val']:
        if phase == 'train':
            net.train()
        else:
            net.eval()

        epoch_loss = 0.0  # epochの損失和
        epoch_corrects = 0  # epochの正解数

        # データローダーからミニバッチを取り出すループ
        for batch in tqdm((dataloaders_dict[phase])):
            # batchはTextとLableの辞書オブジェクト

            # GPUが使えるならGPUにデータを送る
            inputs = batch.Text[0].to(device)  # 文章
            labels = batch.Label.to(device)  # ラベル

            # optimizerを初期化
            optimizer.zero_grad()

            # 順伝搬（forward）計算
            with torch.set_grad_enabled(phase == 'train'):

                # mask作成
                input_pad = 1  # 単語のIDにおいて、'<pad>': 
                input_mask = (inputs != input_pad)

                # Transformerに入力
                outputs, _, _ = net(inputs, input_mask)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                # 訓練時はバックプロパゲーション
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # 結果の計算
                epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                # 正解数の合計を更新
                epoch_corrects += torch.sum(preds == labels.data)

        # epochごとのlossと正解率
        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
        if phase == 'train':
            train_loss = epoch_loss
        else:
            val_loss = epoch_loss
        print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs, phase, epoch_loss, epoch_acc))
    # ep_end
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss, val_loss])

    if epoch % 1 == 0:
        plot_history(csv_path, png_path)

save_path = './weights.pth'
torch.save(net.state_dict(), save_path)

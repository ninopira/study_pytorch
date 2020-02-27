# パッケージのimport
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models

from tqdm import tqdm

from utils.dataloader_image_classification import ImageTransform, make_datapath_list, HymenopteraDataset

# GPUが使えるかを確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)

# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# DatasetとDataLoaderの作成

# アリとハチの画像へのファイルパスのリストを作成する
train_list = make_datapath_list(phase="train")
val_list = make_datapath_list(phase="val")

# Datasetを作成する
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
train_dataset = HymenopteraDataset(
    file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')

val_dataset = HymenopteraDataset(
    file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')

# DataLoaderを作成する
batch_size = 32

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False)

# 辞書オブジェクトにまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}


# 学習済みのVGG-16モデルをロード

# VGG-16モデルのインスタンスを生成
use_pretrained = True  # 学習済みのパラメータを使用
net = models.vgg16(pretrained=use_pretrained)

# VGG16の最後の出力層の出力ユニットをアリとハチの2つに付け替える
net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

# 訓練モードに設定
net.train()

print('ネットワーク設定完了：学習済みの重みをロードし、訓練モードに設定しました')

# 損失関数の設定
criterion = nn.CrossEntropyLoss()

# ファインチューニングで学習させるパラメータを、変数params_to_updateの1～3に格納する

params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

# 学習させる層のパラメータ名を指定
update_param_names_1 = ["features"]
update_param_names_2 = ["classifier.0.weight",
                        "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

# パラメータごとに各リストに格納する
for name, param in net.named_parameters():
    if update_param_names_1[0] in name:
        param.requires_grad = True
        params_to_update_1.append(param)
        print("params_to_update_1に格納：", name)

    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        print("params_to_update_2に格納：", name)

    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        print("params_to_update_3に格納：", name)

    else:
        param.requires_grad = False
        print("勾配計算なし。学習しない：", name)

# 最適化手法の設定
optimizer = optim.SGD([
    {'params': params_to_update_1, 'lr': 1e-4},
    {'params': params_to_update_2, 'lr': 5e-4},
    {'params': params_to_update_3, 'lr': 1e-3}
], momentum=0.9)

num_epochs=2
# 初期設定
# GPUが使えるかを確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)

# ネットワークをGPUへ
net.to(device)

# ネットワークがある程度固定であれば、高速化させる
torch.backends.cudnn.benchmark = True

# epochのループ
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-------------')

    # epochごとの訓練と検証のループ
    for phase in ['train', 'val']:
        print(phase)
        if phase == 'train':
            net.train()  # モデルを訓練モードに
        else:
            net.eval()   # モデルを検証モードに

        epoch_loss = 0.0  # epochの損失和
        epoch_corrects = 0  # epochの正解数

        # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
        if (epoch == 0) and (phase == 'train'):
            continue

        # データローダーからミニバッチを取り出すループ
        for inputs, labels in tqdm(dataloaders_dict[phase]):

            # GPUが使えるならGPUにデータを送る
            inputs = inputs.to(device)
            labels = labels.to(device)

            # optimizerを初期化
            optimizer.zero_grad()

            # 順伝搬（forward）計算
            with torch.set_grad_enabled(phase == 'train'):
                outputs = net(inputs)
                loss = criterion(outputs, labels)  # 損失を計算
                _, preds = torch.max(outputs, 1)  # ラベルを予測

                # 訓練時はバックプロパゲーション
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # 結果の計算
                epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                # 正解数の合計を更新
                epoch_corrects += torch.sum(preds == labels.data)

        # epochごとのlossと正解率を表示
        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = epoch_corrects.double(
        ) / len(dataloaders_dict[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))
# PyTorchのネットワークパラメータの保存
save_path = './weights_fine_tuning.pth'
torch.save(net.state_dict(), save_path)


# PyTorchのネットワークパラメータのロード
load_path = './weights_fine_tuning.pth'
load_weights = torch.load(load_path)
net.load_state_dict(load_weights)

# GPU上で保存された重みをCPU上でロードする場合
load_weights = torch.load(load_path, map_location={'cuda:0': 'cpu'})
net.load_state_dict(load_weights)







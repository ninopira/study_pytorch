# パッケージのimport
import csv
import random
import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image
from tqdm import tqdm

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)


# 可視化
def plot_history(csv_path, png_path):
    df = pd.read_csv(csv_path)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df['epoch'], df['g_train'], label='g_train')
    ax.plot(df['epoch'], df['d_train'], label='d_train')
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    fig.savefig(png_path)

###########################
# アーキテクチャの構築
###########################
class Generator(nn.Module):
    # 乱数を元に画像を出力
    def __init__(self, z_dim=20, image_size=64):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, image_size * 8, kernel_size=4, stride=1),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(image_size * 8, image_size * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
                nn.ConvTranspose2d(image_size * 4, image_size * 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(image_size * 2),
                nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(image_size * 2, image_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True))

        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh())

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)
        return out


class Discriminator(nn.Module):
    # 数字に見える見えないを学習
    def __init__(self, z_dim=20, image_size=64):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, image_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(image_size, image_size*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1,inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(image_size*2, image_size*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(image_size*4, image_size*8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.last = nn.Conv2d(image_size*8, 1, kernel_size=4, stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)
        return out

# # 動作確認
# print('G')
# G = Generator(z_dim=20, image_size=64)
# print(G)
# # 入力する乱数
# input_z = torch.randn(1, 20)
# # テンソルサイズを(1, 20, 1, 1)に変形
# input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
# print(input_z.shape)
# # 偽画像を出力
# fake_images = G(input_z)
# print(fake_images.shape)
# img_transformed = fake_images[0][0].detach().numpy()
# plt.imshow(img_transformed, 'gray')
# plt.show()

# print('D')
# # 動作確認
# D = Discriminator(z_dim=20, image_size=64)
# print(D)
# # 偽画像を生成
# input_z = torch.randn(1, 20)
# input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
# fake_images = G(input_z)
# # 偽画像をDに入力
# d_out = D(fake_images)
# # 2値分類なのでdigmoid
# print(nn.Sigmoid()(d_out))

###########################
# dataloader
###########################
def make_datapath_list():
    """学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。 """

    train_img_list = list()  # 画像ファイルパスを格納

    for img_idx in range(200):
        img_path = "./data/img_78/img_7_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

        img_path = "./data/img_78/img_8_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

    return train_img_list


class ImageTransform():
    """画像の前処理クラス"""

    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)


# datasetの作成
class GAN_Img_Dataset(data.Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        return img_transformed


# dataloaderの作成
train_img_list = make_datapath_list()
train_dataset = GAN_Img_Dataset(
    file_list=train_img_list,
    transform=ImageTransform(mean=(0.5,), std=(0.5,))
)
batch_size = 64
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 動作確認
# print('Data_Loader')
# batch_iter = iter(train_dataloader)
# imges = next(batch_iter)
# print(imges.shape)

###########################
# predict
###########################
def generate(epoch):
    batch_size = 8
    z_dim = 20
    fixed_z = torch.randn(batch_size, z_dim)
    fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

    # 画像生成
    fake_images = G(fixed_z.to(device))

    # # 訓練データ
    # batch_iterator = iter(train_dataloader)  # イテレータに変換
    # imges = next(batch_iterator)  # 1番目の要素を取り出す

    # 出力
    fig = plt.figure(figsize=(15, 6))
    for i in range(0, 5):
        # 上段に訓練データを
        # plt.subplot(2, 5, i+1)
        # plt.imshow(imges[i][0].cpu().detach().numpy(), 'gray')

        # # 下段に生成データを表示する
        # plt.subplot(2, 5, 5+i+1)
        # plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')
        plt.subplot(1, 5, i+1)
        plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')
    fig.savefig('generate_{}ep.png'.format(epoch))





###########################
# learning
###########################
G = Generator()
D = Discriminator()


# 重みの初期化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


G.apply(weights_init)
D.apply(weights_init)

# 最適化手法の設定
g_lr, d_lr = 0.0001, 0.0004
beta1, beta2 = 0.0, 0.9
g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])


# loss. logtをかけて、xエントロピー
criterion = nn.BCEWithLogitsLoss(reduction='mean')

z_dim = 20
mini_batch_size = 64

G.to(device)
D.to(device)

G.train()  # モデルを訓練モードに
D.train()  # モデルを訓練モードに

torch.backends.cudnn.benchmark = True

num_train_imgs = len(train_dataloader.dataset)
# train_dataset.__len__

# 学習
num_epochs = 500
iteration = 1
logs = []
d_loss_means = []
g_loss_means = []
csv_path = './history.csv'
png_path = './history.png'
with open(csv_path, 'w') as f:
    writer = csv.writer(f)
    header = ['epoch', 'g_train', 'd_train']
    writer.writerow(header)
for epoch in range(num_epochs):
    # 開始時刻を保存
    t_epoch_start = time.time()
    epoch_g_loss = 0.0  # epochの損失和
    epoch_d_loss = 0.0  # epochの損失和

    print('-------------')
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('-------------')

    for imgs in tqdm(train_dataloader):
        # Dの学習
        # バッチサイズが0だとBNでコケるの
        if imgs.size()[0] == 1:
            continue
        imgs = imgs.to(device)
        mini_batch_size = imgs.size()[0]
        label_real = torch.full((mini_batch_size,), 1).to(device)
        label_fake = torch.full((mini_batch_size,), 0).to(device)
        # 存在する(真)画像を入れたときのDの出力値
        d_out_real = D(imgs)

        # 入力を乱数でGを通して、画像を生成
        input_z = torch.randn(mini_batch_size, z_dim).to(device)
        input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
        fake_images = G(input_z)

        # Gが生成した偽画像をDを元に本物かの出力値
        d_out_fake = D(fake_images)

        # lossの計算
        d_loss_real = criterion(d_out_real.view(-1), label_real)
        d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
        d_loss = d_loss_real + d_loss_fake

        # バックプロパゲーション
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        d_loss.backward()
        d_optimizer.step()

        # Gの学習
        # 偽の画像を生成して判定
        input_z = torch.randn(mini_batch_size, z_dim).to(device)
        input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
        fake_images = G(input_z)
        d_out_fake = D(fake_images)

        # 誤差を計算
        g_loss = criterion(d_out_fake.view(-1), label_real)

        # バックプロパゲーション
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # --------------------
        # 3. 記録
        # --------------------
        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()
        iteration += 1

    # epochのphaseごとのlossと正解率
    t_epoch_finish = time.time()
    d_loss_mean = epoch_d_loss/batch_size
    g_loss_mean = epoch_g_loss/batch_size
    d_loss_means.append(d_loss_mean)
    g_loss_means.append(g_loss_mean)
    print('-------------')
    print('epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f}'.format(epoch+1, d_loss_mean, g_loss_mean))
    print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        data = {'epoch': epoch+1, 'g_train': g_loss_mean, 'd_train':d_loss_mean}
        writer.writerow([epoch, g_loss_mean, d_loss_mean])

    if epoch % 5 == 0:
        plot_history(csv_path, png_path)
        generate(epoch)
    t_epoch_start = time.time()

print('learning_done')
plot_history(csv_path, png_path)

# PyTorchのネットワークパラメータの保存
save_path = './weights_G.pth'
torch.save(G.state_dict(), save_path)
save_path = './weights_D.pth'
torch.save(D.state_dict(), save_path)



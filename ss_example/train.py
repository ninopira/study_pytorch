'''
学習済みモデルをencoderとしたセマンティックセグメンテーションモデルによる学習
https://github.com/qubvel/segmentation_models.pytorch
https://github.com/qubvel/segmentation_models.pytorch/blob/master/HALLOFFAME.md

'''
import csv
import os

import albumentations as albu
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import segmentation_models_pytorch as smp
import torch

from dataset import CarDataset

cpu = False


DATA_DIR = './data/CamVid/'
# load repo with data if it is not exists
if not os.path.exists(DATA_DIR):
    print('Loading data...')
    os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
    print('Done!')
else:
    print('{}_is_exit'.format(DATA_DIR))
img_dir_path = os.path.join(DATA_DIR, 'train')
ano_dir_path = os.path.join(DATA_DIR, 'trainannot')
val_img_dir_path = os.path.join(DATA_DIR, 'val')
val_ano_dir_path = os.path.join(DATA_DIR, 'valannot')
result_dir_path = './result'
os.makedirs(result_dir_path, exist_ok=True)

# cpu or gpu
if cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('use_device: {}'.format(device))


###########################
# (1)アーキテクチャの構築
###########################
# モデルのパラメータ
num_class = 1  # 複数クラスのssの場合は3や4など2にするとsigmoidもssoftmaxも回らないので注意
# 学習済みモデルの種類
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
# Activation should be callable/sigmoid/softmax/logsoftmax/None
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/base/modules.py#L76
ACTIVATION = 'sigmoid'

# モデルのアーキテクチャの構築
print('building_model...')
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=num_class,
    activation=ACTIVATION,
)
# モデルに入力する際に必要な前処理
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

###########################
# (2)Datasetの作成
###########################
# train用augumentation
# https://github.com/albumentations-team/albumentations を使用。ここに定義されている関数を書き足していけばok
# Note: shapeを32の倍数にしないと回らない
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.5,
            rotate_limit=0,
            shift_limit=0.1,
            p=1,
            border_mode=0),
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),  # padding
        albu.RandomCrop(height=320, width=320, always_apply=True),
    ]
    return albu.Compose(train_transform)


# tvalid用augumentation
def get_validation_augmentation():
    '''
    32の倍数になるようにpadding
    '''
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


# モデルの入力に変換する関数。
#  smp.encoders.get_preprocessing_fnで取得した関数を入力にする
def get_preprocessing(preprocessing_fn):
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization functio
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# datasetの作成
print('building_Dataset...')

train_dataset = CarDataset(
    img_dir_path=img_dir_path,
    ano_dir_path=ano_dir_path,
    aug=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn)
)
val_dataset = CarDataset(
    img_dir_path=val_img_dir_path,
    ano_dir_path=val_ano_dir_path,
    aug=get_validation_augmentation(),  # val用のaugを用いていることに注意
    preprocessing=get_preprocessing(preprocessing_fn)
)

print('num_train_images: {}'.format(train_dataset.__len__()))
print('num_val_images: {}'.format(val_dataset.__len__()))

# 以下のような関数で実際にモデルに入力する画像を確かめられる
# for i in range(3):
#     print(i)
#     plt.imshow(augmented_dataset.__getitem__(i)[0])
#     plt.show()
#     plt.imshow(augmented_dataset.__getitem__(i)[1])
#     plt.show()

###########################
# (3)DataLoaderの作成
###########################
print('building_DataLoader...')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)


###########################
# (4)学習
###########################
# loss / optimizerの設定
# https://github.com/qubvel/segmentation_models.pytorch/blob/6b79c831fa3d8df17a5c4e207ebf304bbf42c094/segmentation_models_pytorch/utils/losses.py
loss = smp.utils.losses.DiceLoss()
# https://github.com/qubvel/segmentation_models.pytorch/blob/6b79c831fa3d8df17a5c4e207ebf304bbf42c094/segmentation_models_pytorch/utils/metrics.py
metrics = [smp.utils.metrics.IoU(threshold=0.5)]
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

# Runnerクラスの作成
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=device,
    verbose=True,
)
valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=device,
    verbose=True,
)


# 学習曲線関連
png_loss_path = os.path.join(result_dir_path, 'history_loss.png')
csv_loss_path = os.path.join(result_dir_path, 'history_loss.csv')
with open(csv_loss_path, 'w') as f:
    writer = csv.writer(f)
    header = ['epoch', 'train_dice_loss', 'val_dice_loss']
    writer.writerow(header)

png_metric_path = os.path.join(result_dir_path, 'history_metric.png')
csv_metric_path = os.path.join(result_dir_path, 'history_metric.csv')
with open(csv_metric_path, 'w') as f:
    writer = csv.writer(f)
    header = ['epoch', 'train_iou', 'val_iou']
    writer.writerow(header)


# 可視化
def plot_history(csv_path, png_path):
    df = pd.read_csv(csv_path)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    train_col = [col for col in df.columns if 'train' in col][0]
    val_col = [col for col in df.columns if 'val' in col][0]
    ax.plot(df['epoch'], df[train_col])
    ax.plot(df['epoch'], df[val_col])
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel('epoch')
    ax.legend()
    fig.savefig(png_path)

model_weights_dir = os.path.join(result_dir_path, 'weights')
os.makedirs(model_weights_dir, exist_ok=True)

# train model
num_ep = 100
max_score = 0
for i in range(num_ep):
    print('\nEpoch: {}'.format(i+1))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # loss / metricの保存
    with open(csv_loss_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([i+1, train_logs['dice_loss'], valid_logs['dice_loss']])
    with open(csv_metric_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([i+1, train_logs['iou_score'], valid_logs['iou_score']])
    # 学習曲線
    if (i+1) % 5 == 0:
        plot_history(csv_loss_path, png_loss_path)
        plot_history(csv_metric_path, png_metric_path)
        model_path = os.path.join(model_weights_dir, 'ep{}_model.pth'.format(i+1))
        torch.save(model, model_path)

    # ベストモデルの保存
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        best_ep = (i+1)
        best_model = model

print('training_done')
print('best_ep: {}'.format(best_ep))
best_model_path = os.path.join(result_dir_path, 'best_model_ep{}.pth'.format(best_ep))
torch.save(model, best_model_path)

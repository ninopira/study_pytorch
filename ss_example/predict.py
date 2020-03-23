import os

import albumentations as albu
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm

from dataset import CarDataset

# cpu or gpu(gpuで学習したモデルをcpuで予測も可能)
cpu = False
if cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('use_device: {}'.format(device))

# モデルのパラメータ
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'

# input_dir_path
DATA_DIR = './data/CamVid/'
img_dir_path = os.path.join(DATA_DIR, 'test')
ano_dir_path = os.path.join(DATA_DIR, 'testannot')
result_dir_path = './result'
model_name = 'best_model_ep100.pth'
model_name = 'weights/ep5_model.pth'


# train済モデルの読み込み
print('load_model...')
model_path = os.path.join(result_dir_path, model_name)
model = torch.load(model_path)
model.to(device)


# Datasetの作成
def get_test_augmentation():
    '''
    32の倍数になるようにpadding
    '''
    test_transform = [
        albu.Resize(384, 480, always_apply=True)
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

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

test_dataset = CarDataset(
    img_dir_path=img_dir_path,
    ano_dir_path=ano_dir_path,
    aug=get_test_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    predict=True
)
print('num_test_images: {}'.format(test_dataset.__len__()))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


# 可視化
def visualize(png_path, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        # import ipdb; ipdb.set_trace()
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
    plt.show()
    plt.savefig(png_path)


predict_dir = os.path.join(result_dir_path, 'predict')
os.makedirs(predict_dir, exist_ok=True)
for image, image_ori, ano, image_name in tqdm(test_loader):
    image_name = os.path.splitext(image_name[0])[0]  # ('0000047.png',) -> 0000047
    pred = model(image.float().to(device))

    png_path = os.path.join(predict_dir, 'predcit_{}.png'.format(image_name))
    visualize(
        **{
            'png_path': png_path,
            'image': image_ori.squeeze(0).cpu().numpy().round(),
            'ground_truth_mask': ano.cpu().detach().numpy().round().squeeze(),
            'predicted_mask': pred.squeeze().cpu().detach().numpy().round()
        }
    )
print('done')

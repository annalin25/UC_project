from LoadDataset import *
from sklearn.metrics import accuracy_score
import random
import warnings
from tqdm import tqdm
from model.resnet import *
from model.resnet2p1 import *
from model.cnn import *
import numpy as np
import json
import torchvision.transforms as transforms
import time

warnings.filterwarnings(action='ignore')


def train(model, device, train_loader, optimizer, epoch):
    # set model as training mode
    model.train()

    train_total_loss = 0
    train_classification_loss = 0

    SEA_epoch_accuracy = 0

    N_count = 0  # counting total trained sample in one epoch

    tbar = tqdm(train_loader)

    for batch_idx, (X, y) in enumerate(tbar):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )
        N_count += X.size(0)

        output_classification = model(X)
        CLloss = H_loss(output_classification, y, device)

        Totalloss = CLloss

        # to compute accuracy
        y_pred = torch.max(output_classification, 1)[1]

        y_detection = torch.ones(y.size(0)).to(device)
        y_detection[y != 4] = 0

        step_score_classification = accuracy_score(y.cpu(), y_pred.cpu())

        SEA_epoch_accuracy += step_score_classification
        train_total_loss += Totalloss.item()
        train_classification_loss += CLloss.item()

        optimizer.zero_grad()
        Totalloss.backward()
        optimizer.step()

        # show information
        log_interval = int(len(train_loader) / 4)
        if log_interval != 0 and (batch_idx + 1) % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Totaloss:{:.4f}  CLloss:{:.4f} | CL_Accu:{:.2f}%'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
                    Totalloss.item(), CLloss.item(),
                    100 * step_score_classification))

    train_total_loss /= len(train_loader)
    train_classification_loss /= len(train_loader)
    SEA_epoch_accuracy /= len(train_loader)

    return train_total_loss, train_classification_loss, SEA_epoch_accuracy


def generate_model(type='ResNet', layer=18):
    if type == 'ResNet2p1':
        # 18
        if layer == 18:
            model_ = ResNet2p1(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_classes=5).to(device)
        # 50
        else:
            model_ = ResNet2p1(Bottleneck, [3, 4, 6, 3], get_inplanes(), n_classes=5).to(device)
    elif type == 'ResNet':
        # 18
        if layer == 18:
            model_ = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_classes=5).to(device)
        # 50
        else:
            model_ = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), n_classes=5).to(device)
    else:
        model_ = ConvNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_classes=5).to(device)

    return model_


def test(model, device, test_loader):
    model.eval()

    gt = []
    prediction = []

    with torch.no_grad():
        print("\nSEA Evaluation")
        for X, y in tqdm(test_loader):
            X, y = X.to(device), y.to(device).view(-1, )
            output_classification = model(X)

            y_pred = torch.max(output_classification, 1)[1]
            y_gt = y
            gt.extend(y_gt)
            prediction.extend(y_pred)

    gt = torch.stack(gt, dim=0).cpu().data.squeeze().numpy()
    prediction = torch.stack(prediction, dim=0).cpu().data.squeeze().numpy()
    test_score = accuracy_score(gt, prediction)

    print('Test set ({:d} samples) >> SEA: {:.2f}%'.format(len(gt), 100 * test_score))
    MAE = float(sum(abs(gt - prediction))) / len(gt)
    print('MAE = ', MAE)


def Dataset(place='LA', partition=False):
    params = {'batch_size': 7, 'shuffle': True, 'num_workers': 4,
              'pin_memory': True} if use_cuda else {}
    params_test = {'batch_size': 7, 'shuffle': False, 'num_workers': 4,
                   'pin_memory': True} if use_cuda else {}

    # LA
    if place == 'LA':
        train_path = 'DATASET/DATASET/train/'
        test_path = 'DATASET/DATASET/test/'
    # Korea
    else:
        train_path = 'DATASET/Seoul_train/'
        test_path = 'DATASET/Seoul_test/'

    if partition:
        train_loader = data.DataLoader(DatasetDeviance(train_path, selected_frames, transform=transform,
                                                       partition=True, place=place), **params)
    else:
        train_loader = data.DataLoader(DatasetDeviance(train_path, selected_frames, transform=transform,
                                                       place=place), **params)
    test_loader = data.DataLoader(DatasetDeviance(test_path, selected_frames, transform=transform,
                                                  place=place), **params_test)

    return train_loader, test_loader


def fine_tune(model):
    # load saved model
    model.load_state_dict(torch.load('saved_model/50_ResNet_20230107_170840'))

    # freeze layers
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)
    model.to(device)

    # fine tune
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 100

    train_loss = []
    train_acc = []
    for epoch in range(epochs):
        train_losses, CE_losses, train_CLscores = train(model, device, train_loader, optimizer, epoch)
        train_loss.append(CE_losses)
        train_acc.append(train_CLscores)
    print('train_loss:', train_loss)
    print('train_acc:', train_acc)

    torch.save(model.state_dict(), os.path.join('saved_model/FT_50_' + str(layer) + '_' + model_type))


if __name__ == '__main__':

    begin_frame, end_frame, skip_frame = 0, 12, 1
    selected_frames = np.arange(end_frame)

    transform = transforms.Compose([transforms.CenterCrop((480, 640)),
                                    transforms.Resize([224, 224]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    use_cuda = torch.cuda.is_available()  # check if GPU exists
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # -------------------load dataset---------------------

    train_loader, test_loader = Dataset(place='LA', partition=False)

    # -----------------model------------------------------
    # model_type = 'ResNet2p1','ResNet','CNNet'
    model_type = 'ResNet'
    layer = 50
    model = generate_model(type=model_type, layer=layer)

    # ----------------train model----------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 100

    train_loss = []
    train_acc = []
    for epoch in range(epochs):
        train_losses, CE_losses, train_CLscores = train(model, device, train_loader, optimizer, epoch)
        train_loss.append(CE_losses)
        train_acc.append(train_CLscores)
    print('train_loss:', train_loss)
    print('train_acc:', train_acc)

    # save model
    now = time.localtime()
    date = "%04d%02d%02d_%02d%02d%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    torch.save(model.state_dict(), os.path.join('saved_model/CNN_' + str(layer) + '_' + model_type + '_' + date))

    # ---------------finetune model--------------------------
    # fine_tune(model)

    # ----------------test model------------------------------
    test(model, device, test_loader)

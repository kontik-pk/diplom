from torch.utils.data import Dataset, DataLoader
from matplotlib import colors, pyplot as plt
from sklearn.preprocessing import LabelEncoder
from efficientnet_pytorch import EfficientNet
from PIL import Image
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
import torch.nn as nn
import pickle
import torchvision
from torchvision import transforms, models


def create_labels(label_path):
  l = pd.read_csv(label_path, sep = ',')
  p = pd.DataFrame()
  p['id'] = l.iloc[:, 0] + '.jpg'
  p['breed'] = l.iloc[:, 1]

  d = {}
  for i in range(0, p.shape[0]):
    d[p.iloc[i, 0]] = p.iloc[i, 1]

  return d

class DogsDataset(Dataset):
   def __init__(self, files, mode, label_path = '/content/dogs/labels.csv', transform=None):
     super().__init__()
     # список файлов для загрузки
     self.files = sorted(files)
     self.labels_dict = create_labels(label_path)

     DATA_MODES = ['train', 'val', 'test']
     # режим работы
     self.mode = mode
     if self.mode not in DATA_MODES:
       print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
       raise NameError
     self.len_ = len(self.files)

     self.label_encoder = LabelEncoder()
     if self.mode != 'test':
       self.labels = [self.labels_dict[path.name] for path in self.files]
       #self.labels = [path.parent.name for path in self.files]
       self.label_encoder.fit(self.labels)
       with open('label_encoder.pkl', 'wb') as le_dump_file:
         pickle.dump(self.label_encoder, le_dump_file)

     if (transform == None):
       self.transform_train = transforms.Compose([transforms.ToTensor(), 
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
     else:

       self.transform_train = transform

   def __len__(self):
       return self.len_

   def load_sample(self, file):
       image = Image.open(file)
       image.load()
       return image
   def _prepare_sample(self, image):
     image = image.resize((224, 224))
     return np.array(image)

   def __getitem__(self, index):

     transform_train = self.transform_train
    
     transform_test = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
     x = self.load_sample(self.files[index])
     x = self._prepare_sample(x)
     x = np.array(x / 255, dtype='float32')
     if self.mode == 'test':
         return transform_test(x) #тестовую не меняем
     else:
         if self.mode == 'train':
           x= transform_train(x)
         else:
           x=transform_test(x)
         label = self.labels[index]
         label_id = self.label_encoder.transform([label])
         y = label_id.item()
         return x, y



def imshow(inp, title=None, plt_ax=plt, default=False):
 inp = inp.numpy().transpose((1, 2, 0))
 mean = np.array([0.485, 0.456, 0.406])
 std = np.array([0.229, 0.224, 0.225])
 inp = std * inp + mean
 inp = np.clip(inp, 0, 1)
 plt_ax.imshow(inp)
 if title is not None:
   plt_ax.set_title(title)
 plt_ax.grid(False)


def fit_epoch(model, train_loader, criterion, optimizer, device = torch.device("cuda")):

  running_loss = 0.0
  running_corrects = 0
  processed_data = 0
  for inputs, labels in train_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    preds = torch.argmax(outputs, 1)
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)
    processed_data += inputs.size(0)

  train_loss = running_loss / processed_data
  train_acc = running_corrects.cpu().numpy() / processed_data
  return train_loss, train_acc


def eval_epoch(model, val_loader, criterion, device = torch.device("cuda")):
  model.eval()
  running_loss = 0.0
  running_corrects = 0
  processed_size = 0
  for inputs, labels in val_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.set_grad_enabled(False):
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      preds = torch.argmax(outputs, 1)
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)
    processed_size += inputs.size(0)
  val_loss = running_loss / processed_size
  val_acc = running_corrects.double() / processed_size
  return val_loss, val_acc

def train(train_dataset, val_dataset, model, opt, epochs, batch_size, name):
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
  history = []
  final_acc = 0
  log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
  val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"
  with tqdm(desc="epoch", total=epochs) as pbar_outer:
    #opt = torch.optim.AdamW(model.parameters(), lr=0.00005, amsgrad=True)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
      train_loss, train_acc = fit_epoch(model, train_loader, criterion, opt)
      print("loss", train_loss)
      val_loss, val_acc = eval_epoch(model, val_loader, criterion)
      if val_acc > final_acc:
        torch.save({'model_state_dict': model.state_dict(),}, name)
        final_acc = val_acc
        print("Saved model with val acc", val_acc.item())
      history.append((train_loss, train_acc.item(), val_loss, val_acc.item()))
      pbar_outer.update(1)
      tqdm.write(log_template.format(ep=epoch+1, t_loss=train_loss,v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))

  return history

def predict(model, test_loader, device = torch.device("cuda")):
  with torch.no_grad():
    logits = []

    for inputs in test_loader:
      inputs = inputs.to(device)
      model.eval()
      outputs = model(inputs).cpu()
      logits.append(outputs)

    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs

def plot(history, name):

  loss, acc, val_loss, val_acc = zip(*history)
  plt.plot(loss, label="train_loss")
  plt.plot(val_loss, label="val_loss")
  plt.legend(loc='best')
  plt.xlabel("epochs")
  plt.ylabel("loss")
  plt.title(name)
  plt.show()

def make_prediction(name, model, test_files):
  
  checkpoint = torch.load(name)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()

  test_dataset = DogsDataset(test_files, mode='test')
  test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)

  probs = predict(model, test_loader)
  label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))

  result = pd.DataFrame(columns = label_encoder.inverse_transform(np.arange(0, 120)))

  result['id'] = [path.name.split('.')[0] for path in test_files]

  for i in range(len(test_files)):
    result.iloc[i, :-1] = probs[i]
    
  return result



def initialize_and_train_model(model_name, output_num, epochs, batch_size, train_dataset, val_dataset, save_name):
  models_list = ["ResNet152", "VGG19", "DenseNet161", "EfficientNet"]

  if model_name not in models_list:
       print(f"{model_name} is not correct; correct modes: {models_list}")
       raise NameError

  if (model_name == "ResNet152"):
    model = models.resnet152(pretrained=True)

    model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=1000, bias=True),
                             nn.Dropout(0.5),
                             nn.Linear(1000, 120, bias=True))
  elif (model_name == "VGG19"):
    model = models.vgg19(pretrained = True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=120, bias=True)

  elif (model_name == "DenseNet161"):
    model = models.densenet161(pretrained=True)

    model.classifier = nn.Sequential(nn.Linear(2208, 128),
                              nn.BatchNorm1d(128),
                              nn.ReLU(),
                              nn.Dropout(p=0.5),
                              nn.Linear(128, 120))
  
  else:
    model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=120)

  torch.cuda.empty_cache()
  opt = torch.optim.AdamW(model.parameters(), lr=0.00005, amsgrad=True)
  history = train(train_dataset, val_dataset, model.cuda(), opt, epochs, batch_size, save_name)
  plot(history, model_name)
  return model
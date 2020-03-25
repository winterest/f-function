from torch.utils import data
import os
import torch
from torchvision import transforms as T
from scipy import interpolate
from PIL import Image
from random import shuffle
import xml.etree.ElementTree as ET


## Config
img_size = 256
## End of config


class LabeledImageFolder(data.Dataset):
    def __init__(self, root, GT_path,list_img_path,image_size=224,mode='train',augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.root = root
        # GT : Ground Truth
        self.GT_paths = GT_path
        #self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_paths = list_img_path
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0,90,180,270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

    def __getitem__(self, index):
        #img_size = 224
        """Reads an image from a file and preprocesses it and returns."""
        
        image_path = self.image_paths[index]
        filename = image_path.split('_')[-1][:-len(".jpg")]
        #GT_path = self.GT_paths + 'ISIC_' + filename + '_segmentation.png'

        image = Image.open(image_path)
        #GT = Image.open(GT_path)
        annot_fn = self.GT_paths + filename.split('/')[-1] + '.xml'
        tree = ET.parse(annot_fn)
        objs = tree.findall('object')

        #img = Image.open(fn)
        wid = image.width
        hei = image.height

        for ix, obj in enumerate(objs):
            if obj.find('name').text.lower().strip()=='graph':
                bbox = obj.find('bndbox')
                x11 = int(float(bbox.find('xmin').text))
                y11 = int(float(bbox.find('ymin').text))
                x12 = int(float(bbox.find('xmax').text))
                y12 = int(float(bbox.find('ymax').text))

            if obj.find('name').text.lower().strip()=='xypercent':
                xper = obj.find('xper')
                #print(xper.text)
                xper = xper.text.split(' ')
                xper = [int(float(i)*224) for i in xper]

                yper = obj.find('yper')
                #print(yper.text)
                yper = yper.text.split(' ')
                yper = [int(float(i)*224) for i in yper]

        image = image.crop((x11,y11,x12,y12)).resize((img_size,img_size))
        matrix = torch.zeros(img_size,img_size)
        vector = torch.ones(img_size) * (-1)

        f = interpolate.interp1d(xper, yper)
        xnew = list(range(xper[0],xper[-1]+1))
        ynew = f(xnew)
        ynew = [int(i) for i in ynew]

        for n,xn in enumerate(xnew):
            matrix[xn, ynew[n]] = 1
            vector[xn] = ynew[n]

        Transform = []

        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image_t = Transform(image)

        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image_t = Norm_(image_t)
            
        return image_t, vector, matrix, image_path

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader(root_path, image_size, batch_size, split_ratio = 0.99, num_workers=2, mode='train',augmentation_prob=0.4):
    """Builds and returns Dataloader."""
    image_path = root_path+'/JPEGImages/'
    GT_path = root_path+'/Annotations/'

    list_all = list(map(lambda x: os.path.join(image_path, x), os.listdir(image_path)))
    shuffle(list_all)
    num_train = int(split_ratio * len(list_all))
    list_train = list_all[:num_train]
    list_val = list_all[num_train:]

    train_dataset = LabeledImageFolder(root = image_path, GT_path=GT_path, list_img_path=list_train,
                          image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
    train_loader = data.DataLoader(dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers)

    val_dataset = LabeledImageFolder(root = image_path, GT_path=GT_path, list_img_path=list_val,
                          image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
    val_loader = data.DataLoader(dataset=val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers)
    return train_loader, val_loader
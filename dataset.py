import cv2
import numpy as np
import os, glob
import ray
from torch.utils.data import Dataset

def make_patches(img:np.ndarray, p:int)->np.ndarray:
    """
        generate image patches in a serial manner

        - Args
            img (np.ndarray): an image array. (C,H,W) in which H=W=224
            p (int): size of patch
        
        - Returns
            patches (np.ndarray): 2-D array of flattened image patches (each patch has a size of P^2*C)
    """
    patches = np.array([])
    x1, y1 = 0, 0
    _,h,w = img.shape

    for y in range(0, h, p):
        for x in range(0, w, p):
            if (h-y) < p or (w-x) < p:
                break
            
            y1 = min(y+p, h)
            x1 = min(x+p, w)

            tiles = img[:, y:y1, x:x1]

            if patches.size == 0:
                patches = tiles.reshape(1,-1)
                
            else:
                patches = np.vstack([patches, tiles.reshape(1,-1)]) # reshape(-1) or ravel도 사용 가능. flatten은 카피 떠서 쓰는 거

    return patches


class ImageNetDataset(Dataset):
    def __init__(self, data_root:str, p:int, is_train:bool, transforms=None, label_info:str="label.txt"):
        """
        Dataset Class for ViT

        - Args
            data_root (str): a directory data stored.
            p (int): size of patch
            is_train (bool): indicates the instance will be for training dataset or not
            transforms (Transforms): augmentations to be applied for the dataset.
        """
        super(ImageNetDataset, self).__init__()
        
        self.p = p
        self.is_train = is_train
        self.transforms = transforms
        self.label_names = []
        self._label_map = dict()

        if self.is_train:
            self.data_root = os.path.join(data_root, "train")
        else:
            self.data_root = os.path.join(data_root, "val")

        # labels info
        with open(label_info, "r") as f:
            _labels = list(map(lambda x: x.strip().split(" "), f.readlines()))
        
        for cls, cls_n, cls_name in _labels:
            self._label_map[cls] = int(cls_n)
            self.label_names.append(cls_name)

    
        # 아래 데이터 정의하는 부분은 validation에서도 문제는 없으나 fine-tuning할 때는 어떻게 될지 잘 모르겠음.
        self.img_list = glob.glob(f"{self.data_root}/**/*.JPEG", recursive=True) # image-net과 같은 경우에는 확장자가 JPEG only.
        self.labels = list(map(lambda x: self._label_map[x.split("/")[-2]], self.img_list))
        

    def __len__(self):
        return len(self.img_list)

    
    def __getitem__(self, index: int):
        """
        label까지 같이 넘기는 형태로 코드 수정할 것.
        이렇게 하려면 label을 one-hot으로 펼쳐줘야 함... -> 원핫으로 안펼쳐줘도 되는 듯?
        """
        img = cv2.imread(self.img_list[index]) # 이미지 경로 읽어오기 (H,W,C)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # color space 변환 & (C,H,W)로 변경

        if self.transforms:
            img = self.transforms(img)['image'] # albumentations 타입의 transform 적용
        
        # make patches
        patches = make_patches(img, self.p)

        # labels
        label = self.labels[index]

        return patches, label
        


# if __name__ == "__main__":
#     train_root = "./ImageNet"
#     p = 16
#     is_train = True
#     transforms = BaseTransform()

#     dataset = ImageNetDataset(train_root, p, is_train, transforms)

#     from torch.utils.data import DataLoader

#     train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=24)
#     print(next(iter(train_loader)))
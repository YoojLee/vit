import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class BaseTransform(object):
    def __init__(self, downsize=256, crop_size = 224):
        """
        Base transform으로 224로 random crop 정의하고 normalize해주기? 얘 normalize 해주나?
        """
        self.transform = A.Compose(
            [   
                A.Resize(height=downsize, width=downsize),
                A.RandomCrop(height=crop_size, width=crop_size), # always_apply
                A.Normalize(),
                ToTensorV2() # albumentations에서는 normalize 이후에 totensorv2를 사용해줘야 함. (여기서 어차피 c,h,w로 변경)
            ]
        )

    def __call__(self, img):
        """
        얘는 nn.Module을 상속한 게 아니기 때문에 forward를 구현해줘도 __call__과 연결이 되어 있지 않음.
        따라서 BaseTransform과 같은 경우에는 __call__ 메소드를 구현해줘야 함.
        """
        return self.transform(image=img)

class RandAugMixUpAugTransform(object):
    def __init__(self):
        pass

    def __call__(self):
        pass
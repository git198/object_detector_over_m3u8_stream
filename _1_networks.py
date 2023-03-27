
import torch
from PIL import Image
from torchvision import transforms
import urllib
import numpy as np
import cv2

from multiprocessing import Process

from _0_open_m3u8_stream import get_stream


class Deeplabv():
    def __init__(self):

        self.model = torch.hub.load(
            'pytorch/vision:v0.10.0', 
            'deeplabv3_resnet50',
            pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')


    def prepare_for_forward(self, input_image: np.ndarray):
        
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the mode
        return input_batch

    def run_on_image(self, input_image: np.ndarray):
        input_batch = self.prepare_for_forward(input_image)
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
        with torch.no_grad():
            output = self.model(input_batch)['out'][0] 
        return output


    def run_and_plot(self, input_image: np.ndarray):
        output =self.run_on_image(input_image)
        output_predictions = output.argmax(0).cpu().numpy()

        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        display_image = np.take(colors, output_predictions,axis=0)
        display_image = 0.7 * input_image + 0.3*display_image
        display_image  = display_image.astype(np.uint8)
        return display_image

    def short_test(self):
        url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
        try: urllib.URLopener().retrieve(url, filename)
        except: urllib.request.urlretrieve(url, filename)
        input_image = Image.open(filename).convert("RGB")
        out_image = self.run_and_plot(input_image)
        cv2.namedWindow("Deeplabv")
        cv2.imshow(out_image)
        cv2.waitKey(1)
        
        


class YoloV5():
    def __init__(self):
        # Model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


    def short_test(self):
        # Images
        imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images
        results = self.model(imgs)
        results.print()
        results.show()

    def run_and_plot(self,img):
        img_copy = img.copy()
        result = self.model(img_copy)
        result._run(render=True)
        return result.ims[0]
    
class YoloV5Large(YoloV5):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load()

def run_test(model):

    model = model()

    uri = 'https://live.webcamromania.ro/WR065/wrcam_065/playlist.m3u8'    
    stream =  get_stream(uri, './movies_model_test')

    cv2.namedWindow('Display')        
    for frame in stream :
        result = model.run_and_plot(frame)
        
        result = np.vstack((result, frame))[::2,::2,:]
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imshow('Display', result)
        cv2.waitKey(1)
    

if __name__ == "__main__":

    run_test(Deeplabv)
    run_test(YoloV5)
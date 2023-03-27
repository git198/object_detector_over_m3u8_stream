""" 
Run YoloV5 and Deeplabv in paralel using ray. 

There are 4 main process running in parallel:
- RayStream :
    - reads stream from m3u8 using get_stream function
    - dumps frame (720p) to ray global variable
- RayActor (YoloV5 or DeepLabv)
    - loads DNN 
    - forwards curent frame throug the network
    - dumps images with detection/segementation to the asigned output global variable 
- RayViewer 
    - reads current output from YoloV5 and DeepLabv
    - stacks output and scales down images by a factor of 2. 
    - displays result using openCV 

Hard coded variables:
- input stream - Piata Romana: 'https://live.webcamromania.ro/WR065/wrcam_065/playlist.m3u8'
- stream working dir : ./movies
"""
import ray
import time
import numpy as np
import cv2

from _0_open_m3u8_stream import get_stream
from _1_networks import YoloV5, Deeplabv

@ray.remote
class RayGlobalVar:
    def __init__(self):
        self.var = None

    def set_var(self, var):
        self.var = var

    def get_var(self):
        return self.var



@ray.remote
class RayStream:
    def __init__(self, uri, out_folder, num_chunks, global_img, run_flag):
        self.stream = get_stream(uri, out_folder, num_chunks)
        self.run_flag = run_flag
        self.global_img = global_img


    def run_in_loop(self):
        for i, data in enumerate(self.stream):
            ray.get(self.global_img.set_var.remote(data))
            if i ==0: ray.get(self.run_flag.set_var.remote(1))
        ray.get(self.run_flag.set_var.remote(0))


class Loop():
    def __init__(self, run_flag):
        self.run_flag = run_flag

    def run_step(self):
        raise NotImplementedError("")

    def run_in_loop(self):
        # Wait for stream to start
        while not ray.get(self.run_flag.get_var.remote()):
            time.sleep(0.1)

        # Run as along as we have a stream
        while ray.get(self.run_flag.get_var.remote()):
            self.run_step()

    
@ray.remote
class RayLoop(Loop):
    def __init__(self, run_flag,data):
        super().__init__(run_flag)
        self.data = data
    
    def run_step(self):
        print(self.data)


@ray.remote
class RayViewer(Loop):
    def __init__(self, stream_1, stream_2, run_flag):
        super().__init__(run_flag)
        self.stream_1 = stream_1
        self.stream_2 = stream_2
        #self.run_flag = run_flag
        cv2.namedWindow('Viewer')
 
    def run_step(self):
        img = ray.get(self.stream_1.get_var.remote())
        img2 = ray.get(self.stream_2.get_var.remote())
        # plot when we have 2 images 
        if isinstance(img,np.ndarray) and isinstance(img2, np.ndarray):
            
            img = cv2.cvtColor(np.vstack([img,img2]), cv2.COLOR_RGB2BGR)
            cv2.imshow('Viewer', img[::2,::2,:] )
            cv2.waitKey(1)


@ray.remote(num_cpus=2, num_gpus=0.5)
class RayActor(Loop):
    def __init__(self,model_cls, input_data, output_data, run_flag, window_name):
        super().__init__(run_flag)
        self.model = model_cls()
        self.input_data = input_data
        self.output_data = output_data
        self.run_flag = run_flag
        self.window_name = window_name

    
    def run_step(self):
        img = ray.get(self.input_data.get_var.remote())
        img = self.model.run_and_plot(img)
        ray.get(self.output_data.set_var.remote(img))
    

if __name__ == "__main__":

    run_flag = RayGlobalVar.remote()
    in_img = RayGlobalVar.remote()
    out_img1 = RayGlobalVar.remote()
    out_img2 = RayGlobalVar.remote()
  
    uri = 'https://live.webcamromania.ro/WR065/wrcam_065/playlist.m3u8'
    out_folder = './movies_ray'

    stream  = RayStream.remote( uri, out_folder, 10, in_img, run_flag)
    actor_1 = RayActor.remote( YoloV5, in_img, out_img1, run_flag,'Model1')
    actor_2 = RayActor.remote(Deeplabv, in_img, out_img2, run_flag, 'Model2')
    viewer = RayViewer.remote( out_img1, out_img2, run_flag)

    actors = [stream, actor_1,actor_2, viewer]

    ray.get([a.run_in_loop.remote() for a in actors])

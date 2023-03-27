import time 
import os
import multiprocessing
import m3u8
import ffmpeg
import cv2

import numpy as np

def delete_dir(input_dir):
    while os.path.isdir(input_dir):
        os.system(f'rm -rf {input_dir}')

def reset_dir(input_dir):
    delete_dir(input_dir)
    os.makedirs(input_dir)


class M3U8Playlist():
    def __init__(self, uri, ts_queue , num_chunks = 10,):
        self.uri = uri
        self.num_chunks = num_chunks
        self.ts_queue = ts_queue
        self.all_files = set([])
      

    def __call__(self):
        if self.uri is None:
            self.uri = uri

        num_chunks = 0
        while num_chunks < self.num_chunks: 
            m3u8_playlist = m3u8.load(self.uri)
            #print(m3u8_playlist.files)
            new_files = sorted(m3u8_playlist.files)[-1:]
            new_files = set(new_files).difference(self.all_files)
            new_files = list(new_files)
            
            #logging.debug(f"get_movies :  new_files :{new_files}")
            if len(new_files) > 0:
                self.all_files.add(new_files[0])
                num_chunks +=1

                ts_movie =  m3u8_playlist.base_uri + f'/{new_files[0]}' 
                print("-----------> ts movie : ", ts_movie)
                self.ts_queue.put( ts_movie)
            else:
                time.sleep(0.1)

        
        self.ts_queue.put(None)
        print("DONE reading m3u8")

def movie_download(ts_movie, out_folder):
    print("Download :", ts_movie)
    movie_file = ts_movie.split("/")[-1]
    
    out_file = out_folder + f'/{movie_file}'
    #print(out_file)
    os.system( f'wget {ts_movie} -O {out_file} -q')
    #wget.download(ts_movie, out_file)
    while not os.path.isfile(out_file):
        time.sleep(0.01)
    return out_file


def ffmpeg_open(movie_q, pipe, out_folder):
    ts_movie = movie_q.get()
    while ts_movie:  # ts movie to be downloaded 
        print(f"ffmpeg_open {ts_movie}")
        movie = movie_download( ts_movie, out_folder)
        movie = os.path.abspath(movie)
        print("Open Movie: " , movie)
        process = (
            ffmpeg
            .input(movie)
            .video
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(quiet=True, pipe_stdout=True, pipe_stderr=True)
        )   

        while True:
            width = 1920
            height = 1080
            data = process.stdout.read(width * height * 3)
            if len(data)==0:
                break
            in_frame = (
                np.frombuffer(data, np.uint8)
                .reshape([1080, 1920, 3])
            )
            in_frame = cv2.resize( in_frame , (1280, 720))
            pipe.send( in_frame)
        pipe.send(None)
        ts_movie = movie_q.get()

    print("DONE loading local movies")
    return 



def read_process_pipe(pipe_in, pipe_out):
    x = pipe_in.recv()
    frame_counter = 0
    while x is not None:
        frame_counter += 1
        pipe_out.send(x)
        x = pipe_in.recv()
    else:
        print(f"Frame counter {frame_counter}")

def load_movies( ts_queue, pipe_frames, out_folder):

    m1 = ts_queue.get()
    if m1 is  None:
        return

    q1 = multiprocessing.Queue()
    q2 = multiprocessing.Queue()
    pipe1_in, pipe1_out = multiprocessing.Pipe()
    pipe2_in, pipe2_out = multiprocessing.Pipe()
    
    p1 = multiprocessing.Process( target=ffmpeg_open, args=(q1, pipe1_in, out_folder))
    p1.start()
    p2 = multiprocessing.Process( target=ffmpeg_open, args=(q2, pipe2_in, out_folder))
    p2.start()

    q1.put(m1)
    running = True

    while running:
        # ffmpeg open 2
        m2 = ts_queue.get() # get movie to be downloaded
        if m2 is None:
            print('Stop m2')
            q1.put(None)
            q2.put(None)
            running = False
            ts_queue.put(None)
        
        else:
            print("Open {m2}")
            q2.put(m2)

        print("Load second movie ", m1)
        if m1 is not None:
            read_process_pipe( pipe1_out, pipe_frames)


        # ffmpeg open 1
        m1 = ts_queue.get()
        if m1 is None:
            print("Stop m1")
            q1.put(None)
            q2.put(None)
            ts_queue.put(None)
            running = False
        else:
            print("Open m1 ", m1)
            q1.put(m1)

        # read stream 2
        print("Load second movie ", m2)
        if m2 is not None:
            read_process_pipe(pipe2_out, pipe_frames)

    pipe1_out.poll()
    pipe2_out.poll()
    ts_queue.get()
    pipe_frames.send(None)
    print("Waiting to finish parallel reading")
    p1.join()
    p2.join()
    print("process joined")    



def get_stream(uri, out_folder,  chunks = 5):
    reset_dir(out_folder)

    ts_queue = multiprocessing.Queue()   
    frame_in , frame_out = multiprocessing.Pipe()

    f1 =  M3U8Playlist(uri, ts_queue, chunks)
    
    p1 = multiprocessing.Process(target=f1, args=())
    p2 = multiprocessing.Process(
        target = load_movies,
        args =  (ts_queue, frame_in, out_folder)
    )
    p1.start()
    p2.start()

    x = frame_out.recv()
    while isinstance(x, np.ndarray):
        x = frame_out.recv()
        if x is not None:
            yield x

    p1.join()
    p2.join()
    

def display_stream(uri):

    stream = get_stream(uri, './movies_stream_test',  chunks = 5)

    cv2.namedWindow("Stream")
    for x in stream:
        if x is None:
            break
        cv2.imshow('Stream', cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
       
if __name__ == "__main__":
    uri  = 'https://live.webcamromania.ro/WR065/wrcam_065/playlist.m3u8'
    display_stream(uri)
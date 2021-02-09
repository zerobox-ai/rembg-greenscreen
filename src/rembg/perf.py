import moviepy.editor as mpy
from tqdm import tqdm
import torchvision

def get_input_frames():
    
    reader = torchvision.io.VideoReader(filepath, "video")
    
    for frame in next(reader):
        yield frame

if __name__ == "__main__":  
    filepath = "C:\\Users\\tim\\Videos\\test\\tim_scarfe-1608643016102.mp4"

    for image in tqdm(get_input_frames()):
        x=3
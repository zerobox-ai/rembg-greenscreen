from rembg.multiprocessing import parallel_greenscreen

if __name__ == "__main__":

    parallel_greenscreen("/Users/zihao/Desktop/zero/video/group15B_Short.avi", 
        3, 
        1, 
        "u2net_human_seg",
        frame_limit=100)

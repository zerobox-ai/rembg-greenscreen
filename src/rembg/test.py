

from .basic_greenscreen import basic_greenscreen
from .multiprocessing import parallel_greenscreen



if __name__ == '__main__':
   # parallel_greenscreen("C:\\Users\\tim\\Videos\\test\\2021-01-31 14-05-36.mp4", 
    #            worker_nodes = 10, 
    #            cpu_batchsize = 50, 
     #           gpu_batchsize = 20)

    basic_greenscreen(
            "C:\\Users\\tim\\Videos\\test\\2021-01-31 14-05-36.mp4", 
            20)
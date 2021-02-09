
from io import BytesIO
import io
import pathlib
import time
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
from ..u2net.detect import nn_forwardpass_http

image = "C:\\Users\\tim\\Pictures\\Screenshot 2020-11-07 224940.png"

for i in tqdm(range(100)):

    t0 = time.time()
    stream = io.BytesIO()    
    np.save(stream, np.zeros( (30,3,320,320)) )
    stream.seek(0)
    t1 = time.time()
    print( t1-t0, " compress the np array (25,3,320,320)" )

    nn_forwardpass_http( stream )
    print(i)






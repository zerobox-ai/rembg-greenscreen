## Greenscreen Bounding Box

#### Go to project folder to install requirements

```
pip install -r requirements.txt
```
#### Modify input video route in src/test.py
```
parallel_greenscreen(file_path,
                         worker_nodes,
                         gpu_batchsize,
                         model_name,
                         frame_limit=-1,
                         prefetched_batches=4,
                         framerate=-1):
                         
```
*For example: check the first 100 frames
```
parallel_greenscreen("/Users/zihao/Desktop/zero/video/group15B_Short.avi", 
        3, 
        1, 
        "u2net_human_seg",
        frame_limit=100)

```
#### Run
```
python src/test.py
```
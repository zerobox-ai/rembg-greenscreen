# Rembg Virtual Greenscreen Edition (Dr. Tim Scarfe)


Rembg Virtual Greenscreen Edition is a tool to create a green screen matte for videos

<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/ecsplendid/rembg/master/examples/greenscreen.png" width="100%" />
</p>


## Video Virtual Green Screen Edition


[WATCH THE VIDEO DEMONSTRATION/Explainer HERE](https://share.descript.com/view/YTo9QAZU5EC)


* Take any  video file and convert it to an alpha matte to apply for a virtual green screen
* It runs end-to-end non-interactively 
* You need ffmpeg installed and on your path
* There is also a powershell script `./remove-bg.ps1` which will do the job in a manual way i.e. first create frames, then run the `rembg -p ...` command and then run ``ffmpeg`` to create the matte movie. This was my first approach to solve this problem but then I migrated onto just making a new version of rembg.  




Usage;

```
python -m src.rembg.cmd.cli -g "video.mp4"
```

Experimental parallel green screen version;

```
python -m src.rembg.cmd.cli --parallelgreenscreen "video.mp4" --workernodes 3 --gpubatchsize 5 --cpubatchsize 2500
```

Be careful with the default parameters, my 11GB GPU is already pretty much maxed with 3 instances of the NN with 5 image mini batches in forward pass. If you max your RAM (cpu * worker nodes) you will start swapping to disk and that will kill you.  


The command above will produce a `video.matte.mp4` in the same folder


## Architecture / performance log


The first thing I changed in the architecture was as follows; 



<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/ecsplendid/rembg/master/examples/Architecture%20v1.png" width="65%" />
</p>

* Making the calls to the NN more "chunky" rather than chatty i.e. it's possible to send batches of around 25 images through the NN on a GPU with 11GB ram. I assumed this would dramatically increase throughput, but actually it didn't. I changed the architecture to be more pipeline oriented and using lazy evaluation (generators). 
* I also increased throughput by first downsizing all images to as near as possible to the receptive field of the NN i.e. 320^2^3, this will speed up all the quadratic processing steps before and not lose any performance
* I also changed the NN model to be the human segmentation variant, which is significantly better for green screen purposes

The next big problem which I assumed was the root of all the issues was the disk IO, having to write all the frames to the disk. I solved this by streaming frames in from MoviePy and streaming them into FFMPEG using STDIN.

<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/ecsplendid/rembg/master/examples/Architecture%20v2.png" width="65%" />
</p>

* This is a significantly more elegant architecture
* No writing of frames to HDD, although it means you need to start again from scratch if you terminate prematurely
* Because the small input frames are no longer being compressed, the results are noticably better
* I also removed the cutout stage and just return the mask/matte
* I remove a bunch of the expensive pre-processing as it didn't make any difference i.e. ther BGR, normalisation (just use scaling)
* There is still 2 expensive resize steps in there i.e. to square aspect to go into model and back again on the other side. This is unavoidable but at least it's on a low res image (height=320).


Much to my surprise though, even this architecture is getting poor throughput, about 15 frames per second end to end. This is disapointing. 

This is a perf log of the code;

<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/ecsplendid/rembg/master/examples/perf.png" width="65%" />
</p>

Looking at this, there are no super obvious bottlenecks which stick out that we can do anything about, other than the obvious which is to use a smaller NN model. There are lots of expensive quadratic resize operations. We can probably improve some memory allocation and IO stuff. 


<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/ecsplendid/rembg/master/examples/multithreead.png" width="65%" />
</p>

I experimented with creating a distributed processing version with the `multiprocessing` library in python. I was really expecting this to give a big speedup, but, alas not so much. I certainly get good utilisation on the machine now (see image below). The thing which kills it is shifting and serialing all the uncompressed image data between the worker nodes. This can amount to 40GB or so for a single batch. 

<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/ecsplendid/rembg/master/examples/multiprocess.png" width="65%" />
</p>

So the net result is we go from about 10fps->26 which is 2.6x improvement. At least we are nearly at real-time for a 30 fps video! 


<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/ecsplendid/rembg/master/examples/multiproc_enhanced.png" width="65%" />
</p>

I enhanced the multiprocessing architecture a bit:

- To also stream to FFMPEG in parallel while waiting for frames to process. 
- I now write and pass back a mono image which cuts the data IPC by 1/3*1/2
- Cleaned up the batching code to make bugs less likely and fixed a bug where the last batch didnt get processed

<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/ecsplendid/rembg/master/examples/Arch5.png" width="65%" />
</p>

- Made an NN server in a separate process using shared memory
- This is making it super clear what the bottleneck is for me i.e. GPU memory. It takes 0.7 seconds for me to batch 25 images, and 25 is the limit before I run out of memory on 11.3gb card. I think some very high throughput might be possible with this archiecture if you have more GPU ram 
- This architecture also leaves open the possibility to have external GPU servers although currently only shared memory on same machine is implemented

<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/ecsplendid/rembg/master/examples/Arch6.png" width="65%" />
</p>

Some thoughts on vnext above

Please get in touch with me if you have more ideas.  

Some ideas now: 

* Get the worker nodes to stream in from the same video file independently, to save sending all the uncompressed video data over IPC (at least in one direction)
* Even though the mask is mono, I am currently padding it to three channels (only way I could figure out how to get FFMPEG to write it). I could do this on the main thread and not send it over IPC, this would cut the data by a third. 
* Get the workers to send the mask over IPC back to the main thread as a high quality JPG, this would probably cut the size down by 20x at least but lose tiny bit of quality and compute time
* Carefully profile the actual masking code again and make some fine tuning, especially on the object allocation stuff

This is pretty much at the point where I am happy-ish with it i.e. I can leave a lot of videos converting overnight and have confidence they will be ready in the morning. 

 [My Whimsical notes](https://whimsical.com/ffmpeg-virtial-greenscreen-tS2T9uthKdCWhxvBAFUcy) are here

## Important notes

* Don't use VBR videos, it will run forever -- use Handbrake to convert them to CFR


### Usage as a cli


Remove the background from all images in a folder
```bash
rembg -p path/to/inputs
```

Produce a matte from a video
```bash
rembg -g path/to/video
```




### References

- https://arxiv.org/pdf/2005.09007.pdf
- https://github.com/NathanUA/U-2-Net
- https://github.com/pymatting/pymatting

### License

 - Copyright (c) 2020-present [Daniel Gatis](https://github.com/danielgatis)
 - Copyright (c) 2020-present [Dr. Tim Scarfe](https://github.com/ecsplendid)

Licensed under [MIT License](./LICENSE.txt)
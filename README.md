# Rembg Virtual Greenscreen Edition (Dr. Tim Scarfe)


Rembg Virtual Greenscreen Edition is a tool to create a green screen matte for videos

<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/ecsplendid/rembg/master/examples/greenscreen.png" width="100%" />
</p>

## Video Virtual Green Screen Edition

[15th Jan 2021 -- made a new YouTube explainer](https://www.youtube.com/watch?v=4NjqR2vCV_k)

* Take any  video file and convert it to an alpha matte to apply for a virtual green screen
* It runs end-to-end non-interactively 
* You need ffmpeg installed and on your path
* There is also a powershell script `./remove-bg.ps1` which will do the job in a manual way i.e. first create frames, then run the `rembg -p ...` command and then run ``ffmpeg`` to create the matte movie. This was my first approach to solve this problem but then I migrated onto just making a new version of rembg.  

If you have any ideas for speeding this up further, please let us know. We have tried quite a few things at this stage and are a bit stuck on how to proceed from here. See some of the "evolution" in  the [Whimsical notes](https://whimsical.com/ffmpeg-virtial-greenscreen-tS2T9uthKdCWhxvBAFUcy).

Usage;

```
pip install rembg-greenscreen

greenscreen -g "path/video.mp4"
```

Experimental parallel green screen version;

```
greenscreen --parallelgreenscreen "path/video.mp4" --workernodes 3 --gpubatchsize 5
```

The command above will produce a `video.matte.mp4` in the same folder, also works with `mov` and `avi` extensions. Uses ffmpeg under the hood to stream and re-encode the frames into a grayscale matte video. 

Be careful with the default parameters, my 11GB GPU is already pretty much maxed with 3 instances of the NN with 5 image gpu batches in forward pass. 

You can see how much free GPU ram you have with 

```
nvidia-smi
```

## CLI interface

<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://raw.githubusercontent.com/ecsplendid/rembg/master/examples/greenscreen_cli.png" width="65%" />
</p>

## Important notes

* Don't use VBR videos, it will run forever -- use Handbrake to convert them to CFR

### References

- https://arxiv.org/pdf/2005.09007.pdf
- https://github.com/NathanUA/U-2-Net (thanks to these folks for making the semantic segmentation model and publishing online)

### License

 - Copyright (c) 2020-present [Daniel Gatis](https://github.com/danielgatis)
 - Copyright (c) 2020-present [Dr. Tim Scarfe](https://github.com/ecsplendid)
 - Copyright (c) 2020-present [Lucas Nestler](https://github.com/ClashLuke) (Making it go faster and more stuff running on the GPU, thanks Lucas!)

Licensed under [MIT License](./LICENSE.txt)
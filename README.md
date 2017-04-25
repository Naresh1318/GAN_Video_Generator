# DCGAN Video Generator [Under development]
Generates a video using DCGAN.

## Dependencies
* tensorflow - Library for ML.
* keras - Awesome high-level NNs API (We'll tf backend).
* PIL - Used to perform some image procesing.
* ffmpeg - used for converting images to videos.


## Discription
This DCGAN is trained using logograms and can be used to generate a random video using the generator by feeding in a random normal tensor of the required frame size.

## Usage
    python3 generate_video.py

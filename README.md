# AudioUpsampling
## CS231N Final Project

### Introduction:
Audio upsampling, the act of improving the sound quality of an
audio clip from a lower resolution, is a field of research garnering a
modest amount of attention recently. Many approaches work
directly with the lower-quality audio as input, and a few have
ventured into using vision learning techniques to solve this task.
Most approaches, however, do not focus on music, a more
delicate form of audio, nor do they use an in-depth computer
vision learning approach. To this end, we propose a novel
computer vision learning method.

### Problem Statement:
We take music audio in the 8kHz range and convert this WAV-form audio input
into mel spectrograms. We then run this image representation of the audio
information through a convolutional neural net-based model to upsample the
sound quality to 44.1kHz. We have tested models employing a Generative
Adversarial Network (GAN) and a vision transformer (ViT) as an encoder to a
GAN. We output generated mel-spectrograms which represent audio in the
44.1kHz range, which images we can convert into audio for a final qualitative
evaluation of our sound upsampling.

### Dataset:
We collected ~11 hours of dance music with
sample rates of 44.1kHz. The music was
processed by cutting each song into 10 second
clips and down sampled clips to 8kHz. We then
converted each clip to a melSpectrogram. This
gave us a novel data set of dance music with the
ability to compare high and low quality clips
directly on a spectrogram over the course of 10
seconds. Although we use unlicensed audio clips,
this data set is unique and dense enough to train
SOTA upsampling models with. Our novel data
set is a major contribution to upsampling music
clips.

### Methods:
To solve the upsampling task we employed two models: one
a pure GAN architecture with the encoder, residual layers,
and decoder, and the other a vision
transformer which we used to replace the encoder of the
pure GAN. Our pre-processing stage involved taking the audio clips
from our new dataset and running them through a short-time fourier transform,
from which we can compute the mel-scaled spectrogram. Images are then normalized
and run through our architectures and loss is calculated using binary cross
entropy. Our accuracy is determined through the standard peak signal-to-noise
ratio.

### Results:
Our first learning is that the pure GAN architecture performs
better than the vision transformer in terms of accuracy. The
model does not quite reproduce the expected high-resolution
frequencies as we hoped but the pure GAN architecture does
the best job of it. 
The losses from the GAN’s discriminator and generators generally decreased overtime,
the discriminator linearly and the generator eventually. Similar loss results can be seen
from the vision transformer, although the variation in loss from its generator is much
more wider than that of its GAN counterpart. It is also interesting to note that the
generator loss for the vision transformer is higher than the discriminator while the
opposite is true for the GAN.

### Conclusions:
Below we show an example generated mel-spectrogram. The bands can
clearly be seen, but it does not quite resemble what a normal high-frequency
spectrogram should. The color differentiation is due to the normalization of the
input gram in the pre-processing stage, so what we care more about are the
wave amplitudes which are disjointed here.
A major issue we faced was computation power to run our models. Both
architectures, the vision transformer one in particular, are greedy and since we
are dealing with upsampling resolution, they require much memory and long
times to train. We were unable to train our models sufficiently given these
constraints. We had to limit our batch size to two images and reduce our
residual layers from six to two as well. We discussed with the teaching team
ways to optimize speed and implemented them, but the gains were still small.
We predict that in the future, with a better model and more optimized training
flow, we could achieve much higher accuracies and better outputs.

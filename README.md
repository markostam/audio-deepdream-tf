# Deep Dreaming on Audio Spectrograms with Tensorflow

This notebook uses the [tensorflow primer on DeepDreaming](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb) to adapt [Christian Dittmar & Stefan Balke's DeepDreamEffect for Caffe](http://labrosa.ee.columbia.edu/hamr_ismir2015/proceedings/doku.php?id=deepdreameffect) from [HAMR 2015](http://labrosa.ee.columbia.edu/hamr_ismir2015/) for tensorflow. In addition, the loss function for inducing hallucinations in the convnet is edited such that high energy areas of the spectrogram are preserved, thereby avoiding distortion upon resynthesis and increasing musicality.

In essence, this hack converts audio spectrograms into images, where they can be processed by specific layers of a pre-trained convulutional neural network ([Inception v3](https://arxiv.org/abs/1512.00567) trained on [ImageNet](https://arxiv.org/abs/1512.00567)) , and then re-synthesized into audio. 

#### In simpler terms: it allows a a convnet to hallucinate audio effects based on its learned weights.

## Dependencies:
Python 3.5

[TensorFlow](https://github.com/tensorflow) for deep learning.

[Librosa](https://github.com/librosa) for audio DSP.

[numpy](http://www.numpy.org)

### 

![](https://github.com/markostam/audio-deepdream-tf/blob/master/images/thugger_dd.jpg?raw=true "")

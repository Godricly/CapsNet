# A MxNet implementation of Capsule

ToDo:
* capsule interpolation example
* revised ReadMe
* some fancy plots

# Rec Result

input | rec output
------| -------
![alt text][input_im]|![alt text][rec_im]

[input_im]: rec/input_x.png "input image"
[rec_im]: rec/rec_x.png "rec image"

# Interpolation
Each row shows interpolation of one digit alone different channel.  
![alt text][interpolation]

[interpolation]: interpolation.gif "interpolation"


# Reference
[Capsule\_MxNet](https://github.com/AaronLeong/CapsNet_Mxnet) Some codes are cropped from this repo, However I'm concerned with its implementation

[Capsule\_Keras](https://github.com/XifengGuo/CapsNet-Keras) Very nice Keras implemtation which clarifies some concepts. I'm still confused with the stop gradient parts.

[MxNet pull #8787](https://github.com/apache/incubator-mxnet/pull/8787) A Capsule implementation which uses old symbol system.


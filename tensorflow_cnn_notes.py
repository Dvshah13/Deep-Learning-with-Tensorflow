# Convolutional neural networks fall into the field of computer vision.  The basic structure of a CNN is as follows using an example of RBG shapes: Start with a box that has a height of 256 and width of 256, and thickness of (3, RGB).  It has RGB values for every pixel.  The CNN will want to compress the height and width but increase the thickness of my picture.  So at the first convolution the height and width become 128 x 128 but the thickness becomes 16.  We continue this in the next convolution wehre the height and width become 64 x 64 but the thickness becomes 64 as well.  Then at another convolution the height and width become 32 x 32 but the thickness becomes 256.  At the end of the procedure what would be left would become a classifier.  The output will then typically be some kind of array such as [0,0,1,0,0...] for example.  To sum it up, the CNN is going to gradually compress the height and width of the picture and increase the thickness and at the end of this, it turns into a classifier.  You can think of the lingo as though you have these different feature maps and they form a big image, then you take a patch or kernal which is a detached smalled part of the big picture.  This patch/kernal has it's own height and width and thickness.  We detach it to analyze it and then run the convolutions to get an output that becomes a higher patch but smaller area (height, weight decrease but thickness increases).  We use a parameter called the stride to determine how many step or pixels to detach another patch.  For example, it the stride = 1, I detach one patch across every 1 pixel.  If the stride = 2, I detach it across every 2 pixels.  Patch/kernal and stride are very important parameters in the CNN.  The stride will run through and detach and the CNN will compress until you get the output.  Those detachments all contain parts of the information of the entire picture.  There are also, two types of padding.  One called valid padding - this padding makes the picture smaller and same padding - shows the same height and width as the picture.  Besides padding, there is another step called pooling.  The pooling is to, for example, if you want to have stride = 2, by doing that, the information density may be lost or the information in the original picture may be destroyed because the stride is too long.  To handle this issue, we add one more step called pooling.  When we keep the stride = 1, we keep more information in the picture, then we use pooling to decrease the size.  The pooling methods in tensorflow are max pooling and average pooling.  So here is an overall summation of the structure of a CNN:  We start with the image -> then pass that into the convolutional layer where we compress it -> then apply pooling (usually max), convolution + pooling can save more valuable information from the original image -> then pass that back into convolution -> then pooling again, note: these steps can be repeated many times -> then we have the fully connected which is like the normal layer we add before -> then another hidden layer -> lastly it becomes a classifier.  Here is a working example of that:  Start with an image, start the convolution and compress the image, we do this until we can get a full connection with all the part of the image, then we use that as the classifier.  The accuracy of a CNN is much higher then a regular neural net without CNN, in the MNIST data example I coded with a regular net the accuracy was 87% which is pretty low.  Using the CNN I could increase that to 96%.  It gets harder to improve and more time intensive to get the accuracy every percentage point higher.

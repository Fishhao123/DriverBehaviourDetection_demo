# DriverBehaviourDetection_demo
A demostration for detect driver dangerous behaviour from image.
This project utilizes Deep Learning method which was trained with the *Distracted Driver Dataset* provided by Kaggle.

Download the model from [HERE](https://pan.baidu.com/s/1-nAXH4Y3iq5XYcD1DF46dA), the extracting code is *59gu*. Then put it in the *model_weights* folder. 

Use `python demo.py` to run the demo, you need to provide the path of your test image. There are some test pictures in the test_images folder in case you don't know which image to choose.

**Notice that this algorithm only precise with the picture taken in a specific view point in a car (i.e. from right to left) becasue of the single images type provided by the Kaggle Datset.** 

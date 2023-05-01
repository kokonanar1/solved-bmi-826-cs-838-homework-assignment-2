Download Link: https://assignmentchef.com/product/solved-bmi-826-cs-838-homework-assignment-2
<br>
<h1>1           Overview</h1>

This assignment is about using convolutional neural networks for image classification. You will implement, design and train deep convolutional networks for scene recognition using PyTorch, an open source deep learning platform. Moreover, you will take a closer look at the learned network by (1) identifying important image regions for the classification and (2) generating adversarial samples that confuse your model. This assignment is team-based. A team can have up to 3 students.

<h1>2           Setup</h1>

<ul>

 <li>Install Anaconda. We recommend using Conda to manage your packages.</li>

 <li>The following packages are needed: PyTorch (1.0.1 with GPU support), OpenCV3, NumPy, Pillow and TensorboardX. And you are in charge of installing them.</li>

 <li>For the visualization of the results, you will need Tensorboard and TensorFlow (a dependency of Tensorboard). You don’t need TensorFlow-gpu in this case.</li>

 <li>You can debug your code and run experiments on CPUs. However, training a neural network is very expensive on CPUs. We recommend using GPU computing for this project. Please setup your team’s cloud instance. Do remember to shutdown the instance when it is not used!</li>

 <li>You will need to download the MiniPlaces dataset for Part II &amp; III of the project. We have included the downloading script. Run download dataset.sh in the assignment folder. All data will be downloaded under ./data/.</li>

 <li>You will need to fill in the missing code in:</li>

</ul>

<em>./code/student </em><em>code.py</em>

<ul>

 <li>You will need to submit your code, results and a writeup. You can generate the submission once you’ve finished the assignment using:</li>

</ul>

<em>python ./zip </em><em>submission.py</em>

<h1>3           Details</h1>

This assignment has three parts. An autograder will be used to grade some parts of the assignment. Please follow the instructions closely.

<h2>3.1         Understanding Convolutions</h2>

In this part, you will need to implement 2D convolution operation–the fundamental component of deep convolutional neural networks. Specifically, a 2D convolution is defined as

<h3>                                                                Y = W∗<em><sub>S </sub></em>X + B                                                              (1)</h3>

<ul>

 <li><strong>Input: </strong>X is a 2D feature map of size <em>C<sub>i </sub></em>×<em>H<sub>i </sub></em>×<em>W<sub>i </sub></em>(following PyTorch’s convention). <em>H<sub>i </sub></em>and <em>W<sub>i </sub></em>are the height and width of the 2D map and <em>C<sub>i </sub></em>is the input feature channels.</li>

 <li><strong>Weight: </strong>W defines the convolution filters and is of size <em>C<sub>o</sub></em>×<em>C<sub>i</sub></em>×<em>K</em>×<em>K</em>, where <em>K </em>is the kernel size. For this part, we only consider squared filters.</li>

 <li><strong>Stride: </strong>∗<em><sub>S </sub></em>is the convolution operation with stride <em>S</em>. <em>S </em>is the step size of the sliding window when W convolves with X. For this part, we only consider equal stride size along the height and width. W is the parameter that will be learned from data.</li>

 <li><strong>Bias: </strong>B is the bias term of size <em>C<sub>o</sub></em>. <em>b </em>is added to every spatial location <em>H </em>× <em>W </em>after the convolution. Again, B is the parameter that will be learned from data.</li>

 <li><strong>Padding: </strong>Padding is often used before the convolution. Again, we only consider equal padding along all sides of the feature map. A (zero) padding of size <em>P </em>adds zeros-valued features to each side of the 2D map.</li>

 <li><strong>Output: </strong><em>Y </em>is the output feature map of size <em>C<sub>o </sub></em>×<em>H<sub>o </sub></em>×<em>W<sub>o</sub></em>, where <em>H<sub>o </sub></em>= + 1 and</li>

</ul>

<strong>Helper Code: </strong>We have provided you helper functions for the implementation (./code/student code.py). You will need to fill in the missing code in the class <strong>CustomConv2DFunction</strong>. You can use the fold / unfold functions and any matrix / tensor operations provided by PyTorch, except the convolution functions. You do not need to modify the code in the class <strong>CustomConv2d</strong>. This is the module wrapper for your code.

<strong>Requirements: </strong>You will need to implement both the forward and backward propagation for this 2D convolution operation. The implementation should work with any kernel size <em>K</em>, input and output feature channels <em>C<sub>i</sub>/C<sub>o</sub></em>, stride <em>S </em>and padding <em>P</em>. Importantly, your implementation need to compute Y given input X and parameters W and B, and the gradients of and . All derivations of the gradients can be found in our course material, except  (provided). In your write up, please describe your implementation.

<strong>Testing Code: </strong>How can you make sure that your implementation is correct? You can compare your forward / backward propagation results with PyTorch’s own Conv2d implementation. You can also compare your gradients with the numerical gradients. We included a sample testing code in ./code/test conv.py. Please make sure your code can pass the test.

<h2>3.2         Design and Train a Convolutional Neural Network</h2>

In the second part, you will design and train a convolutional neural network for scene classification on MiniPlaces dataset.

<strong>MiniPlaces Dataset: </strong>MiniPlaces is a scene recognition dataset developed by MIT. This dataset has 120K images from 100 scene categories. The categories are mutually exclusive. The dataset is split into 100K images for training, 10K images for validation and 10K for testing. You can download the dataset by running download dataset.sh in the assignment folder. The images and annotations will be located under ./data. We will evaluate top-1/5 accuracy for the performance metric. For more details about the dataset, please refer to their github page <a href="https://github.com/CSAILVision/miniplaces">https://github.com/CSAILVision/miniplaces</a><a href="https://github.com/CSAILVision/miniplaces">.</a>

<strong>Helper Code: </strong>We have provided you helper code for training and testing a deep model (./code/main.py). You will have to run this script many times but you are unlikely to modify this file. For your reference, a simple neural network is implemented by the class <strong>SimpleNet </strong>in ./code/student code.py. You will need to modify this class for this part of the project.

<strong>Requirements: </strong>You will design and train a deep network for scene recognition. You model must be trained from scratch using the training set. No other source of information is allowed, e.g., using labels of the validation set for training, or using model parameters that are learned from ImageNet. This part includes 4 different sections.

<ul>

 <li><strong>Section 0</strong>: Let us start by training our first deep network from scratch! You do not need to write any code in this section–we provide the dataloader and a simple network you can use. You can start by running <em>python ./main.py ../data</em></li>

</ul>

You will need to use GPU computing for this training. And it will take a few hours and give you a model with around 47% top-1 accuracy on the validation set. Do remember to put your training inside a container, e.g., tmux, such that your process won’t get killed when you SSH session is disconnected. You can also use <em>watch -n 0.1 nvidia-smi </em>to get a rough estimation of GPU utilization and memory consumption.

Once the traininng in complete, your best model will be saved as ./models/model best.pth.tar. You can evaluate this model by <em>python ./main.py ../data –resume=../models/model best.pth.tar -e</em>

<ul>

 <li><strong>Section 1</strong>: While waiting for the training of the model, you can read the code and understand the training. Please describe the training process implemented in our code in your writeup. You should address the following questions: Which loss function/optimization method is used? How is the learning rate scheduled? Is there any regularization used? Why is top-K accuracy a good metric for this dataset?</li>

 <li><strong>Section 2</strong>: Let us try to use our own convolution to replace PyTorch’s version and train the model for 10 epochs. This can be done by <em>python ./main.py ../data –epoches=10 –use-custom-conv</em></li>

</ul>

How is your implementation different from PyTorch’s convolution in terms of training memory, speed and convergence rate? Why? Describe your findings in the writeup.

<ul>

 <li><strong>Section 3</strong>: Now let us look at our simple network. The current version is a combination of convolution, ReLU, max pooling and fully connected layers. Your goal is to design a better network for this recognition task. There are a couple of things you can explore here. For example, you can add more convolutional layers [5], yet the model might start to diverge in the training. This divergence can be avoided by adding residual connections [2] and/or batch normalization [3]. You might also want to try the multi-branch architecture in Google Inception networks [7]. You can also tweak the hyper-parameters for training, e.g., learning rate, weight decay, training epochs etc. These hyper-parameters can be passed to main.py in the terminal. <em>You should implement your network in student code.py and call main.py for training</em>. Please justify your design of the model and/or the training, and present your results in the writeup. These results include all training curves and training/validation accuracy.</li>

</ul>

<strong>Monitoring the Training: </strong>All intermediate results during training, including training loss, learning rate, train/validation accuracy are logged into files under

./logs. You can monitor and visualize these variables by using <em>tensorboard –logdir=../logs</em>

We recommend copying the logs folder to a local machine and use Tensorboard locally for the curves. Thus, you can avoid to setup a Tensorboard server on the cloud. Please include the curves of your training loss and train/val accuracy in your writeup. Do these curves look normal to you? Please provide your discussion in the writeup.

<strong>[Bonus] MiniPlaces Challenge: </strong>You can choose to upload your final model and thus participate our MiniPlaces challenge. This challenge will be judged by evaluating your model on a hold-out test set. If you decided to do so, please copy your model best.pth.tar to results folder. To make this challenge a bit more challenging, we do have some constraints for your model. First, your model has to be trained under 4 hours using a K40 GPU on the cloud. We do not have a way to strictly enforce this rule, yet please keep this number in mind. Second, your model (tar file) size has to be smaller than 10MB. As a point of reference, our SimpleNet is only 5.5MB with a top-1 accuracy of 47%. Teams that are ranked top 3 in this challenge will received 2 bonus points (out of the 15pt for this homework assignment). We encourage you to take this challenge.

<h2>3.3         Attention and Adversarial Samples</h2>

In the final part, we will look at attention maps and adversarial samples. They present two critical aspects of deep neural networks: interpretation and robustness, and thus will help you gain insight about these networks.

<strong>Helper Code: </strong>Helper code is provided in ./code/main.py and student code.py for visualizing attention maps and generating adversarial samples. For attention maps, you will need to fill in the missing code in class <strong>GradAttention</strong>. And for adversarial samples, you need to complete the class <strong>PGDAttack</strong>.

<strong>Requirements: </strong>You will implement methods for generating attention maps and adversarial samples

<ul>

 <li><strong>Attention</strong>: Suppose you have a trained model. If you minimize the loss of the predicted label and compute the gradient of the loss w.r.t. the input, the magnitude of a pixel’s gradient indicates how important that pixel is for the decision. You can create a 2D attention map by (1) computing the input gradient by minimizing the loss of the predicted label (most confident prediction); (2) taking the absolute values of the gradients; and (3) pick the maximum values across three channels. This method was discussed in [6]. Once you finished the coding, you can run <em>python ./main.py ../data –resume=../models/model best.pth.tar -e -v </em>This command will evaluate your model using your trained model (assuming model best.pth.tar) and visualize the attention maps. All attention maps will saved under ./logs. Again you can use Tensorboard <em>tensorboard –logdir=../logs</em></li>

</ul>

Now you will see a tab named “Image”. And you can scroll the slide bar on top of the image to see samples from different batches. You can also zoom in the image by clicking on it. Please include and discuss the visualization in your writeup.

<ul>

 <li><strong>Adversarial Samples</strong>: Interestingly, if you you minimize the loss of a wrong label and compute the gradient of the loss w.r.t. the input, you can create adversarial samples that will confuse the model! This was first presented in [1]. Let us use the least confident label as a proxy for the wrong label. And you will implement the Projected Gradient Descent in [4]. Specifically, PGD takes several steps of fast gradient sign method, and each time clip the result to the -neighborhood of the input. You will need to be a bit careful for this implementation. You do not want PyTorch to record your gradient operations in the computation graph. Otherwise, it will create a graph that grows indefinitely over time. Again, you can call main.py once you complete the implementation <em>python ./main.py ../data –resume=../models/model best.pth.tar -a -v </em>This command will generate adversarial samples on the validation set and try to attack your model. And you can see how the accuracy drops (significantly!). Moreover, adversarial samples will be saved in the logs folder. And you can use Tensorboard to check them. This time, you will find tabs “Org Image” and “Adv Image”. Can you see the difference between the original images and the adversarial samples? Please discuss your implementation of PGD and present the results (accuracy drop and adversarial samples) in your writeup.</li>

</ul>

<strong>[Bonus] Adversarial Training: </strong>A deep model should be robust under adversarial samples. A possible solution to build this robustness is using adversarial training, as described in [1, 4]. The key idea is to generate adversarial samples and feed these samples into the network during training. To implement adversarial training, you can attach your PGD to the forward function in the SimpleNet (See the comments in the code for details). Unfortunately, this training can be 10x times more expansive than a normal training. To accelerate this process, you can (1) reduce the number of steps in PGD and (2) reduce the number of epochs in training. Your goal is to show that in comparison to a model using normal training, your model using adversarial training has a better chance to survive adversarial attacks. Please discuss your experimental design, implementation and results in the writeup. Your team will received a maximum of 2 bonus points (out of the 15pt for this homework assignment).

<h1>4           Writeup</h1>

For this assignment, and all other assignments, you must submit a project report in PDF. Every team member should send the same copy of the report. Please clearly identify the contribution of all the team members. In the report you will describe your algorithm and any decisions you made to write your algorithm a particular way. Then you will show and discuss the results of your algorithm. In the case of this project, we have included detailed instructions for the writeup in each part of the project. You can also discuss anything extra you did. Feel free to add any other information you feel is relevant. A good writeup doesn’t just show results, it tries to draw some conclusions from your experiments.

<h1>5           Handing in</h1>

This is very important as you will lose points if you do not follow instructions. Every time after the first that you do not follow instructions, you will lose 5%. The folder you hand in must contain the following:

<ul>

 <li>code/ – directory containing all your code for this assignment</li>

 <li>writeup/ – directory containing your report for this assignment.</li>

 <li>results/ – directory containing your results. Please include your model if you decide to participate in our challenge.</li>

</ul>

<strong>Do not use absolute paths in your code </strong>(e.g. /user/classes/proj1). Your code will break if you use absolute paths and you will lose points because of it. Simply use relative paths as the starter code already does. Do not turn in the data / logs / models folder. Hand in your project as a zip file through Canvas. You can create this zip file using <em>python zip submission.py</em>.

<h1>References</h1>

<ul>

 <li>J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples. In <em>ICLR</em>, 2015.</li>

 <li>He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In <em>CVPR</em>, 2016.</li>

 <li>Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In <em>ICML</em>, 2015.</li>

 <li>Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards deep learning models resistant to adversarial attacks. In <em>ICLR</em>, 2018.</li>

 <li>Simonyan, A. Vedaldi, and A. Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. In <em>ICLR</em>, 2014.</li>

 <li>Simonyan and A. Zisserman. Very deep convolutional networks for largescale image recognition. In <em>ICLR</em>, 2015.</li>

 <li>Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In <em>CVPR</em>, 2015.</li>

</ul>
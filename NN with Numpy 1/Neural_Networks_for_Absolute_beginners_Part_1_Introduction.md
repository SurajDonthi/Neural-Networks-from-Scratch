
# Neural Networks with Numpy for Absolute Beginners - Part 1: Introduction

##### In this tutorial, you will get a brief understanding of what Neural Networks are and how they have been developed. In the end, you will gain a brief intuition as to how the network learns.

Artificial Intelligence has become one of the hottest fields in the current day and most of us willing to dive into this field start off with Neural Networks. But on confronting the math intensive concepts of Neural Networks we just end up learning a few frameworks like Tensorflow, Pytorch etc. for implementing Deep Learning Models. Moreover, just learning these frameworks and not understanding the underlying concepts is like playing with a black box. Whether you want to work in the industry or academia, you wil be working and tweaking these for which you need to have a clear understanding. Both the industry and the academia expect you to have full clarity of these concepts. In this tutrial, I'll make it extremely simple to understand Neural Networks by providing step by step explanation. Also, the math you'll need will be the level of high school.

Let us start with the nemesis of artificial neural networks and gain some inspiration as to how it evolved.

## A little bit into the history of how Neural Networks evolved

The evolution of AI dates to back to 1950 when Alan Turing, the computer genius, came out with the Turing Test to distinguish between a Human and a Robot. He describes that when a machine performs so well, that we humans are not able to distinguish between the response given by a human and a machine, it has passed the Turing Test. Apparently this feat was achieved only in 2012, when a company named Vicarious cracked the captchas. Check out this video below on how Vicarious broke the captchas.

>>>>>>>>>>[![](https://github.com/SurajDonthi/Article-Tutorials/blob/master/NN%20with%20Numpy%201/Images/Vicarious.png)](https://www.youtube.com/watch?v=-H185jPf-7o)

It must be noted that most of the Algorithms that were developed during that period(1950-2000) and now existing, are highly inspired by the working of our brain, the neurons and their structure with how they learn and transfer data. The most popular works include the Perceptron and the Neocognitron - (not covered in this article, but in a future article) based on which the Neural Networks have been developed. 

Now, before you dive into what a perceptron is,  let's make sure you know a bit of all these... Although not necessarily required!

## Prerequisites

What you’ll need to know for the course:
1.   A little bit of Python &
2.   The eagerness to learn Neural Networks.

If you are unsure of which environment to use for implementing this, I recommend [Google Colab](https://colab.research.google.com/). The environment comes with many important packages already installed. Installing new packages and also importing and exporting the data is quite simple. Most of all, it also comes with GPU support. So go ahead and get coding with the platform!

Lastly, this article is directed for those who want to learn about Neural Networks or just Linear Regression. However, there would be an inclination towards Neural Networks!

## A biological Neuron

<p align="center">
<img src="https://docs.google.com/drawings/d/e/2PACX-1vSivgdRoiDD8I1uBa1pUk9uALPbsE4LyoSVJpJkxLbT3DqTN-UwAcn4La9jmADG2u-8Ul5dZmDpwVtw/pub?w=3842&h=1698" width=800>
</p>

The figure above shows a biological neuron. It has *dendrites* that recieve information from neurons. The recieved information is passed on to the *cell body or the nucleus* of the neuron. The *nucleus* is where the information is processed. The processed information is passed on to the next layer of neurons through the *axons*.

Our brain consists of about 100 billion such neurons which communicate through electrochemical signals. Each neuron is connected to 100s and 1000s of other neurons which constantly transmit and recieve signals. When the sum of the signals recieved by a neuron exceeds a set threshold value, the cell is activated (although, it has been speculated that neurons use very complex activations to process the input data) and the signal is further transmitted to other neurons. You'll see that the artificial neuron or the perceptron adopts the same ideology to perform computation and transmit data in the next section.

You know that different regions of our brain are activated (/receptive) for different actions like seeing, hearing, creative thinking and so on. This is because the neurons belonging to a specific region in the brain are trained to process a certain kind of information better and hence get activated when only certain kinds of information is being sent.The figure below gives us a better understanding of the different receptive regions of the brain.

<p align="center">
<img src="https://github.com/SurajDonthi/Article-Tutorials/blob/master/NN%20with%20Numpy%201/Images/brain-regions-areas.gif" width=600>
</p>

It has also been shown through the concept of Neuroplasticity that the different regions of the brain can be rewired to perform totally different tasks. Such as the neurons responsible for touch sensing can be rewired to become sensitive to smell. Check out this great TEDx video below to know more about neuroplasticity.

Similarly, an artificial neuron/perceptron can be trained to recognize some of the most comlplex pattern. Hence, they can be called Universal Function Approximators.

In the next section, we'll explore the working of a perceptron and also gain a mathematical intuition.

>>>>>>>>>>[![](https://img.youtube.com/vi/xzbHtIrb14s/0.jpg)](https://www.youtube.com/watch?v=xzbHtIrb14s)

## Perceptron/Artificial Neuron

<p align="center">
<img src="https://docs.google.com/drawings/d/e/2PACX-1vQi5UtWOScAOVixzrE42U59N2o6ruP8_LgHlTF8fSQH4glqZa6AsbkNxmwWAsYKdcjBmUQSyG5zFCod/pub?w=3720&h=2884" alt="Drawing" width="500"/>
</p>

From the figure, you can observe that the perceptron is a reflection of the biological neuron. The inputs combined with the weights(<img src="http://latex.codecogs.com/gif.latex?w_i" title="w_i" />) are analogous to dendrties. These values are summed and passed through an activation function (like the thresholding function as shown in fig.). This is analogous to the nucleus. Finally, the activated value is transmitted to the next neuron/perceptron which is analogous to the axons.

The latent weights(<img src="http://latex.codecogs.com/gif.latex?w_i" title="w_i" />) multiplied with each input(<img src="http://latex.codecogs.com/gif.latex?x_i" title="x_i" />) depicts the significance of the respective input/feature. Larger the value of a weight, more important is the feature. Hence, the weights are what is learned in a perceptron so as to arrive at the required result. An additional bias(<img src="http://latex.codecogs.com/gif.latex?b" title="b" />, here <img src="http://latex.codecogs.com/gif.latex?w_i" title="w_0" />) is also learned.

Hence, when there are multiple inputs (say <img src="http://latex.codecogs.com/gif.latex?n" title="n" />), the equation can be generalized as follows: 

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?z=w_0&plus;w_1.x_1&plus;w_2.x_2&plus;w_3.x_3&plus;......&plus;w_n.x_n" title="z=w_0+w_1.x_1+w_2.x_2+w_3.x_3+......+w_n.x_n" />
<br>
<img src="http://latex.codecogs.com/gif.latex?\therefore&space;z=\sum_{i=0}^{n}w_i.x_i&space;\qquad&space;\text{where&space;}&space;x_0&space;=&space;1" title="\therefore z=\sum_{i=0}^{n}w_i.x_i \qquad \text{where } x_0 = 1" />
</p>

Finally, the output of summation (assume as <img src="http://latex.codecogs.com/gif.latex?z" title="z" />) is fed to the *thresholding activation function*, where the function outputs <img src="http://latex.codecogs.com/gif.latex?-1&space;\&space;\text{if&space;}&space;z&space;<&space;0&space;\&space;\&&space;\&space;1&space;\&space;\text{if&space;}&space;z&space;\geq&space;0" title="-1 \ \text{if } z < 0 \ \& \ 1 \ \text{if } z \geq 0" />.

### An Example

Let us consider our perceptron to perform as *logic gates* to gain more intuition.

Let's choose an <img src="http://latex.codecogs.com/gif.latex?AND&space;\&space;gate" title="AND \ gate" />. The Truth Table for the <img src="http://latex.codecogs.com/gif.latex?AND&space;\&space;gate" title="AND \ gate" /> is shown below:

<p align="center">
 <img src="https://docs.google.com/drawings/d/e/2PACX-1vTBFWuo0jZqGST_0f-zn_oX9u5zmrFQTXDlAu3SZsiOGycQpshBS1HzyxyNJj5iJ7d3AprYyKzjPfYa/pub?w=1441&h=847" alt="Drawing" width="250"/>
</p>

The perceptron for the <img src="http://latex.codecogs.com/gif.latex?AND&space;\&space;gate" title="AND \ gate" /> can be formed as shown in the figure. It is clear that the perceptron has two inputs (here <img src="http://latex.codecogs.com/gif.latex?x_1=A" title="x_1=A" /> and <img src="http://latex.codecogs.com/gif.latex?x_2=B" title="x_2=B" />)

<p align="center">
 <img src="https://docs.google.com/drawings/d/e/2PACX-1vQW2pQ4tL-XVZ09z_dkHiSmrS9-rkoQe7NZz3JMQ1ybErrA9zpDyWIZZVdKhfYhFmbEk3YpPAlT7hx5/pub?w=2783&h=1836" alt="AND Gate" width="300"/>
</p>
<p align="center">
<img src="http://latex.codecogs.com/gif.latex?\text{Threshold&space;Function,}&space;\qquad&space;y&space;=&space;f(z)&space;=&space;\begin{cases}&space;1,&&space;\text{if&space;}z&space;\geq&space;0.5\\&space;0,&&space;\text{if&space;}&space;z<&space;0.5\\&space;\end{cases}" title="\text{Threshold Function,} \qquad y = f(z) = \begin{cases} 1,& \text{if }z \geq 0.5\\ 0,& \text{if } z< 0.5\\ \end{cases}" />
</p>

We can see that for inputs <img src="http://latex.codecogs.com/gif.latex?x_1,&space;x_2&space;\&space;and&space;\&space;x_0=1," title="x_1, x_2 \ and \ x_0=1," /> setting their weights as 
<p align="center">
<img src="http://latex.codecogs.com/gif.latex?w_0=-0.5," title="w_0=-0.5," />
<br>
<img src="http://latex.codecogs.com/gif.latex?w_1=0.6," title="w_1=0.6," />
<br>
<img src="http://latex.codecogs.com/gif.latex?w_2=0.6" title="w_2=0.6" />
</p>

respectively and keeping the *Threshold function* as the activation function we can arrive at the <img src="http://latex.codecogs.com/gif.latex?AND&space;\&space;Gate" title="AND \ Gate" />.

Now, let's get our hands dirty and codify this and test it out!


```python
def and_perceptron(x1, x2):
    
    w0 = -0.5
    w1 = 0.6
    w2 = 0.6
    
    z = w0 + w1 * x1 + w2 * x2
    
    thresh = lambda x: 1 if x>= 0.5 else 0

    r = thresh(z)
    print(r)
```


```python
>>>and_perceptron(1, 1)
```

    1
    

Similarly for <img src="http://latex.codecogs.com/gif.latex?NOR&space;\&space;Gate" title="NOR \ Gate" /> the Truth Table is,

<p align="center">
<img src="https://docs.google.com/drawings/d/e/2PACX-1vSdobJruUXwaSoQ6y9IscvyZEfBEY7xyE8pGZXtfVF8ADgTUdPuOWEBKKEWhCUJ2MokyJqEM_bkxiz9/pub?w=1438&h=809" alt="Drawing" width="250"/>
</p>

The perceptron for <img src="http://latex.codecogs.com/gif.latex?NOR&space;\&space;Gate" title="NOR \ Gate" /> will be as below:

<p align="center">
<img src="https://docs.google.com/drawings/d/e/2PACX-1vTe0faigDdNNxjlSuc8gBZVY6M5Ew9Mp_F_U_xWVWwsW-KDbJ--8Fq2lUfxT5tYdOukT0Fkv91aXSXh/pub?w=2772&h=1834" alt="NOR Gate" width="300"/>
</p>

You can set the weights as
<p align="center">
<img src="http://latex.codecogs.com/gif.latex?w_0&space;=&space;0.5" title="w_0 = 0.5" />
<br>
<img src="http://latex.codecogs.com/gif.latex?w_1&space;=&space;-0.6" title="w_1 = -0.6" />
<br>
<img src="http://latex.codecogs.com/gif.latex?w_2&space;=&space;-0.6" title="w_2 = -0.6" />
</p>

so that you obtain a <img src="http://latex.codecogs.com/gif.latex?NOR&space;\&space;Gate" title="NOR \ Gate" />.

You can go ahead and implement this in code.


```python
def nor_perceptron(x1, x2):
    
    w0 = 0.5
    w1 = -0.6
    w2 = -0.6
    
    z = w0 + w1 * x1 + w2 * x2
    
    thresh = lambda x: 1 if x>= 0.5 else 0

    r = thresh(z)
    print(r)
```


```python
>>>nor_perceptron(1, 1)
```

    0
    

Here, is the Truth Table for <img src="http://latex.codecogs.com/gif.latex?NAND&space;\&space;Gate" title="NAND \ Gate" />. Go ahead and guess the weights that fits the function and also implement in code.

<p align="center">
<img src="https://docs.google.com/drawings/d/e/2PACX-1vTZtYQeTz7QELabtZ7-zzrGlXi-p-L6dwF9FZl4x9So6hfxCxdNC4ANhCELmnVDix-38PlIOlPLqhul/pub?w=1440&h=915" alt="Drawing" width="250"/>
</p>

## What you are actually calculating...

If you analyse what you were trying to do in the above examples, you will realize that you were actually trying to adjust the values of the weights to obtain the required output.

Lets consider the <img src="http://latex.codecogs.com/gif.latex?NOR&space;\&space;Gate" title="NOR \ Gate" /> example and break it down to very miniscule steps to gain more understanding. 

What you would usually do first is to simply set some values to the weights and observe the result, say

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?w_0&space;=&space;0.4" title="w_0 = 0.4" />
<br>
<img src="http://latex.codecogs.com/gif.latex?w_1&space;=&space;0.7" title="w_1 = 0.7" />
<br>
<img src="http://latex.codecogs.com/gif.latex?w_2&space;=&space;-0.2" title="w_2 = -0.2" />
</p>

Then the output will be as shown in below table:
<p align="center">
<img src="https://docs.google.com/drawings/d/e/2PACX-1vRa_92metML3nIWcHhCTB7AYVoAbvIq3-1Phyixx_l05GJ0IOZ86MoUnIrwhqpxMZRQ2N97FVPIJsY-/pub?w=3288&h=1082" alt="AND Gate" width="700"/>
</p>

So how can you fix the values of weights so that you get the right output?

By intuition, you can easily observe that <img src="http://latex.codecogs.com/gif.latex?w_0" title="w_0" /> must be increased and <img src="http://latex.codecogs.com/gif.latex?w_1" title="w_1" /> and <img src="http://latex.codecogs.com/gif.latex?w_0" title="w_0" /> must be reduced or rather made negative so that you obtain the actual output. But if you breakdown this intuition, you will observe that you are actually finding the difference between the actual output and the predicted output and finally reflecting that on the weights...

This is a very important concept that you will be digging deeper and will  be the core to formulate the ideas behind *gradient descent* and also *backward propagation*.

## Conclusion

While it is true that we learnt only a few small concepts (although very crucial) in this first part of the article,  they will serve as the strong foundation for implementing Neural Networks. Moreover, keeping the article short and sweet will keep you enthused and wanting more.

You can checkout the next part of the article here: 

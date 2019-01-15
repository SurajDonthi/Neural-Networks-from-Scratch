
# Neural Networks with Numpy for Absolute Beginners - Part 1: Introduction

##### In this tutorial, you will get a brief understanding of what Neural Networks are and how they have been developed. In the end, you will gain a brief intuition as to how the network learns.

Artificial Intelligence has become one of the hottest fields in the current day and most of us willing to dive into this field start off with Neural Networks!! 

But on confronting the math intensive concepts of Neural Networks we just end up learning a few frameworks like Tensorflow, Pytorch etc., for implementing Deep Learning Models. 

Moreover, just learning these frameworks and not understanding the underlying concepts is like playing with a black box. 

Whether you want to work in the industry or academia, you will be working, tweaking and playing with the models for which you need to have a clear understanding. Both the industry and the academia expect you to have full clarity of these concepts including the math.

In this series of tutorials, I'll make it extremely simple to understand Neural Networks by providing step by step explanation. Also, the math you'll need will be the level of high school.

Let us start with the nemesis of artificial neural networks and gain some inspiration as to how it evolved.

## A little bit into the history of how Neural Networks evolved

It must be noted that most of the Algorithms for Neural Networks that were developed during the period 1950-2000 and now existing, are highly inspired by the working of our brain, the neurons, their structure and how they learn and transfer data. The most popular works include the Perceptron(1958) and the Neocognitron(1980). These papers were extremely instrumental in unwiring the brain code. They try to mathematically formulate a model of the neural networks in our brain. 

And everything changed after the God Father of AI Geoffrey Hinton formulated the back-propagation algorithm in 1986(That's right! what you are learning is more than 30 years old!).

## A biological Neuron

<p align="center">
<img src="https://docs.google.com/drawings/d/e/2PACX-1vSivgdRoiDD8I1uBa1pUk9uALPbsE4LyoSVJpJkxLbT3DqTN-UwAcn4La9jmADG2u-8Ul5dZmDpwVtw/pub?w=3842&h=1698" width=800>
</p>

The figure above shows a biological neuron. It has *dendrites* that receive information from neurons. The received information is passed on to the *cell body or the nucleus* of the neuron. The *nucleus* is where the information is processed and passed on to the next layer of neurons through *axons*.

Our brain consists of about 100 billion such neurons which communicate through electrochemical signals. Each neuron is connected to 100s and 1000s of other neurons which constantly transmit and receive signals. 

But how can our brain process so much information just by sending electrochemical signals? How can the neurons understand which signal is important and which isn't? How do the neurons know what information to pass forward?

The electrochemical signals consist of strong and weak signals. The strong signals are the ones to dominate which information is important. So only the strong signal or a combination of them pass through the nuclues (the CPU of neurons) and are transmitted to the next set of neurons through the axons.

But how are some signals strong and some signals week?

Well, through millions of years of evolution, the neurons have become sensitive to certain kinds of signals. When the neuron encounters a specific pattern, they get triggered(activated) and as a consequence send strong signals to other neurons and hence the information is transmitted.

Most of us also know that different regions of our brain are activated (/receptive) for different actions like seeing, hearing, creative thinking and so on. This is because the neurons belonging to a specific region in the brain are trained to process a certain kind of information better and hence get activated only when certain kinds of information is being sent. The figure below gives us a better understanding of the different receptive regions of the brain.

<p align="center">
<img src="https://raw.githubusercontent.com/SurajDonthi/Article-Tutorials/master/NN%20with%20Numpy%201/Images/Blausen_0102_Brain_Motor%26Sensory_(flipped).png" width="50%">
</p>

If that is so... can the neurons be made sensitive to a different pattern(i.e., if they have truly become sensitive based on some patterns)?

It has been shown through Neuroplasticity that the different regions of the brain can be rewired to perform totally different tasks. Such as the neurons responsible for touch sensing can be rewired to become sensitive to smell. Check out this great TEDx video below to know more about neuroplasticity.

>>>>>>>>>>[![](https://img.youtube.com/vi/xzbHtIrb14s/0.jpg)](https://www.youtube.com/watch?v=xzbHtIrb14s)

But what is the mechanism by which the neurons become sensitive?

Unfortunately, neuroscientists are still trying to figure that out!!

But fortunately enough, god father of AI Geff has saved the day by inventing back propagation which accomplishes the same task for our Artificial Neurons, i.e., sensitizing them to certain patterns.

In the next section, we'll explore the working of a perceptron and also gain a mathematical intuition.

## Perceptron/Artificial Neuron

<p align="center">
<img src="https://docs.google.com/drawings/d/e/2PACX-1vQi5UtWOScAOVixzrE42U59N2o6ruP8_LgHlTF8fSQH4glqZa6AsbkNxmwWAsYKdcjBmUQSyG5zFCod/pub?w=3720&h=2884" alt="Drawing" width="500"/>
</p>

From the figure, you can observe that the perceptron is a reflection of the biological neuron. The inputs combined with the weights(<img src="http://latex.codecogs.com/gif.latex?w_i" title="w_i" />) are analogous to dendrites. These values are summed and passed through an activation function (like the thresholding function as shown in fig.). This is analogous to the nucleus. Finally, the activated value is transmitted to the next neuron/perceptron which is analogous to the axons.

The latent weights(<img src="http://latex.codecogs.com/gif.latex?w_i" title="w_i" />) multiplied with each input(<img src="http://latex.codecogs.com/gif.latex?x_i" title="x_i" />) depicts the significance(strength) of the respective input signal. Hence, larger the value of a weight, more important is the feature.

You can infer from this architecture that the weights are what is learned in a perceptron so as to arrive at the required result. An additional bias(<img src="http://latex.codecogs.com/gif.latex?b" title="b" />, here <img src="http://latex.codecogs.com/gif.latex?w_0" title="w_0" />) is also learned.

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

So, how can you fix the values of weights so that you get the right output?

By intuition, you can easily observe that <img src="http://latex.codecogs.com/gif.latex?w_0" title="w_0" /> must be increased and <img src="http://latex.codecogs.com/gif.latex?w_1" title="w_1" /> and <img src="http://latex.codecogs.com/gif.latex?w_0" title="w_0" /> must be reduced or rather made negative so that you obtain the actual output. But if you breakdown this intuition, you will observe that you are actually finding the difference between the actual output and the predicted output and finally reflecting that on the weights...

This is a very important concept that you will be digging deeper and will be the core to formulate the ideas behind *gradient descent* and also *backward propagation*.

## What did you learn?

- Neurons must be made sensitive to a pattern in order to recognize it.
- So, similarly, in our perceptron/artificial neuron, <b>the weights are what is to be learnt</b>.

In the later articles you'll fully understand how the weights are trained to recognize patterns and also the different techniques that exist.

As you'll see later, the neural networks are very similar to the structure of biological neural networks.

While it is true that we learnt only a few small concepts (although very crucial) in this first part of the article, they will serve as the strong foundation for implementing Neural Networks. Moreover, I'm keeping this article short and sweet so that too much is information is not dumped at once and will help absorb more!

In the next tutorial, you will learn about <b>Linear Regression</b> (which can otherwise be called a perceptron with linear activation function) in detail and also implement them. The <b>Gradient Descent algorithm which helps learn the weights</b> are described and implemented in detail. Lastly, you'll be able to <b>predict the outcome of an event</b> with the help of Linear Regression. So, head on to the next article to implement it!

You can checkout the next part of the article here: 


---
layout: post
title:  "Obstacles along deep learning evolution: vanishing gradients"
date:   2019-05-09 00:00:00 -0500
comments: true
categories:
---

<style>
body {
text-align: justify}
</style>

The objective of this series is to illustrate some of the past and on-going challenges within deep learning. In the previous blog post, the obstacle discussed was the <a href="../saturation">saturation</a> within activation functions during training. This is a follow-up, so I recommend you take a brief look before continuing. 

<!--
## The challenges of training deep networks
Despite mentioning pre-training as solution for saturation, such technique is not employed as much today. This is due to fact that the properties that pre-training ensures can be attained by a much more efficent strategy which will be explained further on. 

https://arxiv.org/pdf/1504.06825.pdf
-->

### Problem: Vanishing gradients
As mentioned in the previous post, the activation values start to migrate towards the extremities of the sigmoid function in networks with several layers. But what happens to the gradients? Around the 1980's researchers at USC and CMU started to work on algorithms using the gradient of the loss function throughout the layers of the network as a way to adjust each weight with a corresponding "influence" over the error. These gradients were obtained analytically by applying the chain rule from the loss function to the desired weight. This is famously called the backpropagation technique, and has spured the development of neural networks later on. (Cite hinton 1989)

However, once again, as the stack of layers increase, obstacles arise to  hamper performance, and this time this is caused by a phenonmenon known as the Vasinhing/Exploding Gradients. To illustrate it, consider a network with a general cost function and symmetric activation functions $f$ with unit derivative at 0. If we write $a\_{i}$ for the activation vector of layer $i$, and $s\_{i}$ the argument vector of the activation function at layer $i$, we have $s\_{i}=a\_{i}W\_{i}+b_{i}$ and $a\_{i}=f(s\_{i})$.

Consider the following node output and activation:

Consider the following node output and activation:

$$s^{i}=W^{i}a^{i-1}+b_{i}$$

$$a\_{i+1}=g\left(W\_{i}a\_{i-1}+b\_{i}\right)$$

The derivative of the loss function $\mathcal{L}$ in a network with $l$ layers in terms of a weight $W\_{n}$ can be written as:

$$\frac{\partial\mathcal{L}}{\partial W\_{n}}=\frac{\partial s\_{n}}{\partial W\_{n}}\left[\underset{i=n}{\overset{l-1}{\prod}}\frac{\partial a\_{i+1}}{\partial s\_{i}}\frac{\partial s\_{i+1}}{\partial a\_{i+1}}\right]\frac{\partial a\_{l}}{\partial s\_{l}}\frac{\partial\mathcal{L}}{\partial a\_{l}}$$

Defining a linear activation function $g(z\_{i})=z\_{i}$ and $W\_{i}=W$, we obtain:

$$\require{cancel}\frac{\partial\mathcal{L}}{\partial W\_{n}}=\frac{\partial s\_{n}}{\partial W\_{n}}\left[\underset{i=n}{\overset{l-1}{\prod}}\cancelto{1}{\frac{\partial a\_{i+1}}{\partial s\_{i}}}\cancelto{W}{\frac{\partial s\_{i+1}}{\partial a\_{i+1}}}\right]\frac{\partial a\_{l}}{\partial s\_{l}}\frac{\partial\mathcal{L}}{\partial a\_{l}}$$

Thus:

$$\frac{\partial\mathcal{L}}{\partial W\_{n}}=a\_{n-1}W^{n-l}\frac{\partial\mathcal{L}}{\partial a\_{l}}$$


So, in deep configurations, the backpropagation technique may become troublesome as a result of exploding or vanishing gradients. As the cost function is progressively derived in terms of numerically large parameters, these adjustments will likely overshoot, hampering training. Conversely, small contributions propagated through many layers may cause virtually no effect at all:

Moreover, in deep configurations, the backpropagation technique may become troublesome as a result of exploding or vanishing gradients. As the cost function is progressively derived in terms of numerically large parameters, these adjustments will likely overshoot, hampering training. Conversely, small contributions propagated through many layers may cause virtually no effect at all:

If the elements in W greater than one, and with a sufficiently large n-l value, $\frac{\partial\mathcal{L}}{\partial W\_{n}}$ will tend to infinity, i.e. exploding. Conversely, for values less than one, the derivative tends to zero, i.e. vanishing.

For the case of early neural network development, a typical initialization framework presented in Erhan et al. (2009) would have parameters randomly sampled from a uniform distribution centered in zero of $[-1/\sqrt{k};1/\sqrt{k}]$ where k is the size of the previous layer of a fully-connected network. For deep architectures, many nodes would likely be initialized very close to zero, leading to the previously mentioned vanishing gradient problem. Bradley (2010, pg. 25) provides an interesting illustration of this, by analyzing the distribution of weights in the initialization of the network. As the number of layers increased, the weights got significantly peaked around zero, thus evidenciating the cause of the issue.

### Activation functions and initialization methods

In the following course of deep learning development, improvements introduced into networks' training and architecture led to the near abandonment of the very one technique that resurged attention onto the field: unsupervised pre-training. Such advances were a combination of modern activation functions and simpler but sophisticated initialization methods. These also helped to mitigate the vanishing gradient problem. But appart from the ackowledged need for better alternatives to pre-training, what is the problem of using the once well-adopted activation function: the sigmoid? By exploring different functions, it became apparent that the activation was one of the main contributors to the saturation issue, demonstrated through research conducted by Glorot and Bengio (2010)

<img src="./figures/saturation_plot.png" width="100%"/>
<p style="font-size:0.8em;" align="center">Mean (<i>lines</i>) and standard deviation (<i>vertical bars</i>) of sigmoid activation values across layers in a neural network using random initialization. The saturation is detectable in the last layer, where the activation values reach virtually zero. <br>Source: [Glorot and Bengio (2010)](#References)</p>

A technique employed way before such improvements although is goes along with them are the old standardizing of inputs and parameters. LeCun et al. (1998) points out that rescaling the input reducing its mean to 0 and variance to 1 may help training. The first prevents undiserable bias towards particular directions, making it less sensible to the distinc features scales while it also prevents undesirable uniform gradient updates if the majority of the input data where of the same sign. The latter helps to even the contributions from all features, balancing the rate at which the weights connected to the input nodes learn. Additionally it also helps to not saturate too early and does not go straight to zero. Such techniques are not only important on the inputs as they are desirable characteristics for each layer within the network. So, preserving the variance throughout layers during training is a desirable property. Additionally from the backpropagation point of view, the variance of the derivative of the cost function by the weights is equally important to be mantained.

Nevertheless, expanding on [Glorot and Bengio (2010](#References) work, [Xu et al. (2016)](#References) draws theorical clues to indicate that in its linear regime, in comparison with other activation functions, the sigmoid is more prone to escalating the variance of layers througout training.

Considering the following notation:

$$s^i=f\left(s^{i-1}\right)W^i+b^i$$

and 

$$a^{i+1}=f\left(a^{i}W^{i}+b^{i}\right)$$

where $f(s)$ and $s$ are the activation function and its argument vector, respectively; $a$ is the activation value, $W\in\mathbb{R}^{{n\_i\times n\_{i-1}}}$ the weight matrix and $b\in \mathbb{R}^{n_i}$ the bias vector. In a linear regime, the activation function can be modeled as:

$$f(s)=\alpha s+\beta$$

Assuming that, similar to the intialization presented in [eq:1], biases are set to 0, the variance at layer i can be defined as:

$$Var\left[a^{i}\right]=\alpha{{}^2}n_i\sigma\_{i-1}^{2}\left(Var\left(a^{i-1}\right)+\beta{{}^2}I\_{n_i}\right)$$

Likewise, the gradient variance, comprised during the backwards pass, is defined as:

$$Var\left(\frac{\partial cost}{\partial a^{i-1}}\right)=\alpha{{}^2}n_i\sigma\_{i-1}^{2}Var\left(\frac{\partial cost}{\partial a^{i}}\right)$$


According to [Glorot and Bengio (2010)](#References), preserving the variance throuhgout layers and iterations is an indication that the information is flowing without loss. Furthermore, it is evident that as the variance increase, the more the activation values resort to the function's extremes, resulting in saturation. Thus, ideally:

$$Var\left(y^{i}\right)=Var\left(y^{i-1}\right)\text{ and }Var\left(\frac{\partial cost}{\partial y^{i}}\right)=Var\left(\frac{\partial cost}{\partial y^{l-1}}\right)$$

assuming $n\_{i}\sigma\_{i-1}^{2}\approxeq1$ and $n\_{i-1}\sigma\_{i-1}^{2}\approxeq 1$. So, to satisfy this condition $\alpha$ and $\beta$ must be:

$$\alpha=1\text{ and }\beta=0$$

Considering the Taylor expansions of different activation functions:

$$sigmoid(x)=\frac{1}{2}+\frac{x}{4}-\frac{x^{3}}{48}+O\left(x^{5}\right)$$

$$tanh(x)=0+x+\frac{x^{3}}{3}+O\left(x^{5}\right)$$

This approximation indicates, that in the linear regime, the tanh function satisfies the condition $\alpha=1$ while the sigmoid posesses a constant term that may explain the increase in variance throughout the feedfoward propagation. Additionally, due to the significantly small slope in comparison with the tanh, the sigmoid function requires a weight initialization 16 times greater ($\alpha^{2}$) than the tanh to maintain the gradient variance ([Xu et al. (2016)](#References)).


<h1><a name="References"></a>References</h1>
<ul style="font-size:0.8em;">
    <li>Bradley, D. M. (2010). Learning In Modular Systems. PhD Thesis, Carnegie Mellon University.</li>
    <li>Erhan, D., Manzagol, P.A., Bengio, Y., Bengio, S., and Vincent, P. (2009). The Difficulty of Training Deep Architectures and the Effect of Unsupervised Pre-Training. Artificial Intelligence and Statistics, 153-160.</li>
    <li>Glorot, X. and Bengio, Y. (2010). Understanding the diffculty of training deep feedforward neural networks. Proceedings of the Thirteenth International Conference on Artificial Intelligence and
    Statistics , pages 249-256.</li>
    <li>Goodfellow, I., Bengio, Y., and Courville, A. (2016). Deep Learning. MIT Press.</li>
    <li>LeCun, Y. A., Bottou, L., Orr, G. B., and Müller, K.-R. (1998). Efficient BackProp. Neural Networks: Tricks of the Trade, 9-48.</li>
    <li>Shanmugamani, R. (2018). Deep Learning for Computer Vision.</li>
    <li>Xu, B., Huang, R., and Li, M. (2016). Revise Saturated Activation Functions.</li>
</ul>


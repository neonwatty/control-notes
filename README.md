# Degree Six | Blog [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Machine%20Learning%20Refined:%20notes,%20exercises,%20and%20Jupyter%20notebooks%20&url=https://github.com/jermwatt/dgsix_blog)

Python demos and Jupyter notebooks associated with blogposts on [blog.dgsix.com](https://blog.dgsix.com) 

<img align="right" src="html/gifs/book.png" height="420">
<strong>Publisher:</strong> Cambridge University Press <br><br>
<strong>First edition:</strong> November 2016 <br>
<strong>Second edition:</strong> January 2020 (expected) <br><br><br>




# Table of contents

- [A little sampler first](#a-little-sampler-first)
- [What is in this book?](#what-is-in-this-book)
- [Who is this book for?](#who-is-this-book-for)
- [What is in the repo?](#what-is-in-the-repo)
- [Notes](#notes)
- [Installation](#installation)
- [Creators](#creators)

<br><br><br>

## A little sampler first

[(Back to top)](#table-of-contents)

Many machine learning concepts - like convergence of an algorithm, evolution of a model from an underfitting one all the way to an overfitting model, etc. - can be illustrated and intuited best using animations (as opposed to static figures). You'll find a large number of both images and animated videos here - which you can modify yourself too via the raw Jupyter notebook version of these notes.  Here are just a few examples:<br><br>



<br><br><br>
## What is in this book?

[(Back to top)](#table-of-contents)

We believe that understanding machine learning is impossible without having a firm grasp of its underlying mathematical machiney. But we also believe that the bulk of learning the subject takes place when learners "get their hands dirty" and code things up for themselves. **That's why in this book we discuss both how to derive machine learnig models mathematically and how to implement them from scratch**  (using `numpy`, `matplotlib`, and `autograd` libraries) - and yes, this includes multi-layer neural networks as well!
<br><br><br>


## Who is this book for?

[(Back to top)](#table-of-contents)

This text aims to bridge the existing gap between **practicality** and **rigor** in machine learning education, in a market saturated with books that are either mathematically rigorous but not practical, or vice versa. Conventional textbooks usually place little to no emphasis on coding, leaving the reader struggling to put what they learned into practice. On the other hand the more hands-on books in the market typically lack rigor, leaving machine learning a 'black box' to the reader.

If you're looking for a practical yet rigorous treatment of machine learning, then this book is for you. 
<br><br><br>


## What is in the repo?

[(Back to top)](#table-of-contents)

### 1. Interatcive html notes
These notes - listed [here](#notes) - served as an early draft for the second edition of the text. You can also find them in the `notes` directory. Here's an example: <br><br>

<p align="center"><img src="html/gifs/html.gif" width="70%" height="auto"></p>
<br>

### 2. Accompanying Jupyter notebooks (used to create the html notes) 
Feel free to take a peek under the hood, tweak the models, explore new datasets, etc. Here's an example: <br><br>

<p align="center"><img src="html/gifs/ipynb.gif" width="65%" height="auto"></p>
<br>

### 3. Coding exercises (1st edition)

In the `exercises` directory you can find starting wrappers for coding exercises from the first edition of the text in `Python` and `MATLAB`. Exercises for the 2nd edition will be added soon.
<br><br><br>


## Topics
[(Back to top)](#table-of-contents)

### Dynamic Systems


<img src="posts/moving_averages/images/moving_avg.png" width="300px" height="auto"> | <img src="posts/one_dimensional_convolutions/images/conv.png" width="300px" height="auto"> | <img src="posts/dynamic_systems_limited_memory/images/ds.png" width="300px" height="auto"> 
---|---|---
[Moving averages](https://blog.dgsix.com/posts/moving_averages/Moving_averages.html) | [1D convolutions](https://blog.dgsix.com/posts/one_dimensional_convolutions/One_dimensional_convolutions.html) | [Dynamic systems with unlimited memory](https://blog.dgsix.com/posts/dynamic_systems_unlimited_memory/dynamic_systems_unlimited_memory.html)

<br><br>

<img src="posts/dynamic_systems_limited_memory/images/ds.png" width="300px" height="auto"> | <img src="posts/markov_chains/images/markov.png" width="300px" height="auto"> | <img src="posts/dynamic_systems_unlimited_memory/images/riemann_v2.png" width="300px" height="auto"> 
---|---|---
[Dynamic systems with limited memory](https://blog.dgsix.com/posts/dynamic_systems_limited_memory/dynamic_systems_limited_memory.html) | [Markov chains and stochastic recurrence relations](https://blog.dgsix.com/posts/markov_chains/Markov_chains.html) | [Dynamic systems with unlimited memory](https://blog.dgsix.com/posts/dynamic_systems_unlimited_memory/dynamic_systems_unlimited_memory.html)

<br><br>




 

## Installation
[(Back to top)](#table-of-contents)

To successfully run the Jupyter notebooks contained in this repo we highly recommend downloading the [Anaconda Python 3 distribution](https://www.anaconda.com/download/#macos). Many of these notebooks also employ the Automatic Differentiator [autograd](https://github.com/HIPS/autograd) which can be installed by typing the following command at your terminal
      
      pip install autograd
      
With minor adjustment users can also run these notebooks using the GPU/TPU extended version of autograd [JAX](https://github.com/google/jax).<br><br><br>


## Creators 

[(Back to top)](#table-of-contents)

This repository is in active development by [Jeremy Watt](mailto:jeremy@dgsix.com) and [Reza Borhani](mailto:reza@dgsix.com) - please do not hesitate to reach out with comments, questions, typos, etc.
-

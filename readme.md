# Geologic AI

Neural networks that run on geologic timescales.

## Why?

Conceptually neural networks are very simple, but without much math background 
they can be hard to understand at first. This project aims to introduce
neural networks starting with an object oriented model and only basic math
in a code first, math second manner. Performance is very slow but is easy to 
step through with a debugger.

## Prerequisites

If you've ever used your browser's javascript debugger you'll probably pick up
this code just fine. When we start getting into the math we'll link to other
resources to dive deeper.

## Chapter 1

Neural networks are a huge category in machine learning but they all share
a couple things in common.

1. They're made up of a graph of nodes and edges
2. They learn by training against input/output examples

Everything else is a design choice. This makes neural networks very flexible
and useful for a many problems, but choosing a good network design often 
takes experimentation and is an open field of study for many problems.

### Recognizing Digits

We'll start our journey with a seemingly simple problem, recognizing hand 
written digits. Let's imagine we've been challenged to write a program to 
do this. If neural networks and other machine learning algorithms weren't 
a thing yet, how might we do it?

One way could be to make a template for each digit, like a stencil. Then
we could overlay each of our templates on the input to see how well they
match up. Ink inside the template is good and ink outside it is bad. We 
can give each template a score for how well it matches the input and 
then pick the one with the best score as our answer.

Now, how should we come up with our templates? Since there's only 10 digits
we could design them by hand without too much effort, but we could also
gather a bunch of sample images of digits and label them. If we average 
the samples for each digit together we'll end up with templates that 
match all of our samples pretty well and probably better than if we 
designed them ourselves.

We also need to decide on a format for the input images and the templates. 
Let's say that we'll size our images to 28x28 pixels and encode them as
greyscale values 0 - 255. Spoiler alert, there's already a standard dataset
used for this problem, the 
[MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist/),
that contains 60k digit images, 50k for training and 10k for validation.
Except, thinking ahead a bit, 8-bit unsigned integers might not be ideal
for averaging our template pixels like we decided earlier, so let's use
 floats with a target range from 0-1.

Now we should be able to write some code to generate our templates and
also to predict what digit an image is, [like this](./js/naive.js).

How well does our little template matching algorithm do? Well, better
than a random guess, but it still leaves a lot to be desired with ~70%
accuracy for our 10k validation images.

We could improve on our algorithm with other techniques. Maybe we have more 
than one template for each digit to cover different styles, like for 7 with or 
without the crossbar, but where do we stop? Or maybe we shift and rotate the 
template to find the optimal placement over the image. Or maybe we can decode 
the pixels into strokes and figure out the digit from those? That sounds like a 
lot of work though and a lot of manual trial and error. What if the template 
idea is good but there's a better way to generate them than averaging our 
training samples?

The thing about our averaged templates that's less than ideal is that we
never check how well the templates are working as we go; there's no "learning"
happening. We can use our training data to check our templates as we go! If
they give us the right answer, then great, but what do we do when the answer
is wrong?
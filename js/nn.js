/**
 * A fully connected feedforward neural network.
 * 
 * This type of neural network is made up of layers of neurons. Each layer gets
 * its input from the previous layer and feeds its output to the next layer. The
 * neurons in each layer take input from all the neurons in the previous layer.
 * 
 * The output of the last layer is the output of the network.
 * 
 * The network is trained using a supervised learning technique called backpropagation.
 * Supervised means that the network is given the correct answer during training. It
 * uses the correct answer to calculate the error of it's output and adjust the weights
 * and biases of the neurons to reduce the error. The algorithm used to adjust the weights
 * and biases is called stochastic gradient descent.
 * 
 * @see {@link https://en.wikipedia.org/wiki/Feedforward_neural_network}
 * @see {@link https://en.wikipedia.org/wiki/Backpropagation}
 * @see {@link https://en.wikipedia.org/wiki/Stochastic_gradient_descent}
 */
export class NeuralNetwork {
  /**
   * The layers of the neural network.
   * @type {Layer[]}
   */
  layers = [];

  /**
   * The mean squared error of the neural network.
   * @type {number}
   */
  meanSquaredError = 0;

  /**
   * Creates a new instance of the NeuralNetwork class.
   * 
   * @param {...number} layerSizes - The sizes of the layers in the neural network.
   * @example
   * // create a network with 3 layers: 784 input, 16 hidden, 11 output
   * const network = new NeuralNetwork(784, 16, 11);
   */
  constructor(...layerSizes) {
    // The input layer is implicit.
    // It's size determines the number of inputs to the next layer.
    for (let i = 1; i < layerSizes.length; i++) {
      this.layers.push(new Layer(layerSizes[i], layerSizes[i - 1]));
    }
  }

  /**
   * The last layer in the neural network.
   * @type {Layer}
   */
  get outputLayer() {
    return this.layers[this.layers.length - 1];
  }

  /**
   * Feeds forward the inputs through the neural network and calculates the outputs.
   * 
   * @param {Float64Array} inputs - The input values.
   * @returns {number[]} - The output values of the neural network.
   */
  feedForward(inputs) {
    for (const layer of this.layers) {
      // the output of the current layer becomes the input of the next layer
      inputs = layer.feedForward(inputs);
    }
    return inputs;
  }

  /**
   * Backpropagates the error through the neural network.
   * 
   * @param {number[]} targetOutputs - The target output values from the training data.
   * @returns {number} - The mean squared error of this iteration.
   *                     This tells us how well the network is learning.
   *                     The lower the error, the better the network is performing.
   *                     @see {@link https://en.wikipedia.org/wiki/Mean_squared_error}
   */
  backpropagate(targetOutputs) {
    let squaredErrorSum = 0;
    // We process the layers in reverse order. The ouput layer
    // uses the target outputs, while the hidden layers use the
    // weights and deltas from the next layer.
    for (let i = 0; i < targetOutputs.length; i++) {
      const neuron = this.outputLayer.neurons[i];
      const target = targetOutputs[i];
      // Intuitively, the error of the output layer is the difference
      // between the target output and the actual output.
      // When the output is close to the target, the error is small.
      // The sign of the error tells us which direction to adjust the weights.
      const error = target - neuron.output;
      neuron.backpropagate(error);
      // Mean squared error is a simple way to measure the error.
      // Mathematically, it's referred to as a loss or cost function.
      // There are many other loss functions to choose from.
      // Ours is a quadratic loss function.
      // @see {@link https://en.wikipedia.org/wiki/Loss_function#Quadratic_loss_function}
      squaredErrorSum += error * error;
    }

    for (let i = this.layers.length - 2; i >= 0; i--) {
      this.layers[i].backpropagate(this.layers[i + 1]);
    }
    return squaredErrorSum / targetOutputs.length;
  }

  /**
   * Updates the weights and biases of the neural network.
   * 
   * @param {number} learningRate - The learning rate.
   */
  updateWeights(learningRate) {
    for (const layer of this.layers) {
      layer.updateWeights(learningRate);
    }
  }

  /**
   * Trains the neural network using the given inputs and target outputs.
   * 
   * @param {import("./mnist").LabelledInputData[]} inputs - The input values.
   * @param {number} [learningRate=1] - The higher the learning rate, the faster the network learns.
   *                                However, if it's too high, the network may not converge.
   * @param {number} [epochs=1] - The number of rounds of training to do over all the inputs.
   *                              More epochs can improve accuracy, but can also lead to overfitting.
   * @param {number} [batchSize=1] - The number of samples to process before updating the weights.
   *                                 Larger batch sizes can speed up training, but may be less accurate.
   * @returns {Generator<NeuralNetwork, void, void>} - A generator that yields the network after each epoch.
   */
  *train(inputs, learningRate = 1, epochs = 1, batchSize = 1) {
    console.time("Training time");
    const trainingData = inputs.map(({ input, label }) => ({
      input,
      // Transform the label into an array representing the desired output layer values.
      output: Array.from({ length: this.outputLayer.size }, (_, i) =>
        i === label ? 1 : 0
      ),
    }));
    for (let i = 1; i <= epochs; i++) {
      console.time(`Epoch ${i}/${epochs}`);
      const epoch = trainingData
        .concat(
          noise(
            trainingData[0].input.length,
            trainingData[0].output.length,
            trainingData.length / 10
          )
        )
        .sort(() => Math.random() - 0.5);
      let error = 0;
      for (let j = 0; j < epoch.length; j += batchSize) {
        const batch = epoch.slice(j, j + batchSize);
        for (const { input, output } of batch) {
          this.feedForward(input);
          error += this.backpropagate(output);
        }
        this.updateWeights(learningRate);
      }
      this.meanSquaredError = error / epoch.length;
      console.timeEnd(`Epoch ${i}/${epochs}`);
      yield this;
    }
    console.timeEnd("Training time");
  }
}

/**
 * A layer of neurons in a neural network.
 */
class Layer {
  /**
   * The unique identifier of the layer.
   * @type {string}
   */
  id = '';

  /**
   * The neurons in the layer.
   * @type {Neuron[]}
   */
  neurons = [];

  /**
   * The output values of the layer.
   * @type {Float64Array}
   */
  outputs = new Float64Array(0);

  /**
   * Creates a new instance of the Layer class.
   * 
   * @param {number} neuronCount - The number of neurons in the layer.
   * @param {number} inputCount - The number of inputs per neuron.
   */
  constructor(neuronCount, inputCount) {
    this.id = `l-${crypto.getRandomValues(new Uint32Array(1))[0].toString(16)}`;
    this.neurons = Array.from(
      { length: neuronCount },
      () => new Neuron(inputCount)
    );
    this.outputs = new Float64Array(neuronCount);
  }

  get size() {
    return this.neurons.length;
  }

  /**
   * Feeds forward the inputs through the layer and calculates the outputs.
   * 
   * @param {Float64Array} inputs - The input values.
   * @returns {Float64Array} - The output values of the layer.
   */
  feedForward(inputs) {
    for (let i = 0; i < this.neurons.length; i++) {
      this.outputs[i] = this.neurons[i].feedForward(inputs);
    }
    return this.outputs;
  }

  /**
   * Backpropagates the error through the layer.
   * 
   * @param {Layer} nextLayer - The next layer in the neural network.
   */
  backpropagate(nextLayer) {
    for (let i = 0; i < this.neurons.length; i++) {
      // The error for this neuron is the sum of the error of all 
      // the weights that took the output of this neuron as input.
      let error = 0;
      for (const neuron of nextLayer.neurons) {
        error += neuron.errorForWeight(i);
      }
      this.neurons[i].backpropagate(error);
    }
  }

  /**
   * Updates the weights and biases of the neurons in the layer.
   * 
   * @param {number} learningRate - The learning rate.
   */
  updateWeights(learningRate) {
    for (const neuron of this.neurons) {
      neuron.updateWeights(learningRate);
    }
  }
}

/**
 * A single neuron in a neural network.
 */
class Neuron {
  /**
   * The unique identifier of the neuron.
   * @type {string}
   */
  id = '';

  /**
   * The inputs to the neuron.
   * Each value is a number between 0 and 1.
   *
   * @type {number[]}
   */
  inputs = [];

  /**
   * The output value of the neuron.
   * It's value is between 0 and 1.
   *
   * @type {number}
   */
  output = 0;

  /**
   * The weights used to calculate the output value.
   * There is one weight per input value.
   * These are updated during training to improve the accuracy of the network.
   * A larger weight value makes it's corresponding input value more influential
   * in the output calculation.
   *
   * @type {Float64Array}
   */
  weights = [];

  /**
   * The bias of the neuron.
   * It is added to the weighted sum of the inputs to influence the output value
   * and is updated during training. A larger bias value makes it easier for the
   * neuron to fire and produce an output value closer to 1.
   *
   * @type {number}
   */
  bias = 0;

  /**
   * The error of the neuron. It is used to calculate a delta value for each
   * weight and bias during backpropagation. The delta value represents the
   * magnitude and direction (+/-) that the weight should be changed to reduce
   * the error. It is part of the gradient descent algorithm.
   *
   * The deltaBiasSum and deltaWeightSums below are used to accumulate the deltas
   * over multiple samples in a batch before updating the weights. These sums are
   * divided by the number of samples in the batch to get the average delta.
   *
   * Batches are used to speed up training.
   *
   * @type {number}
   */
  error = 0;

  /**
   * The sum of the deltas for the bias during a batch.
   * The bias delta is proportional to the error of the neuron and the derivative
   * of the activation function (sigmoid in this case), i.e. the slope of the curve.
   *
   * These deltas are summed over a batch of samples before updating the bias.
   *
   * @type {number}
   */
  deltaBiasSum = 0;

  /**
   * The sum of the deltas for the weights during a batch.
   * The weight delta is proportional to the error of the neuron, the derivative
   * of the activation function, and the input value.
   *
   * These deltas are summed over a batch of samples before updating the weights.
   *
   * @type {Float64Array}
   */
  deltaWeightSums = [];

  /**
   * The number of samples in the batch.
   * It is incremented during backpropagation and
   * reset to 0 after the weights are updated.
   *
   * @type {number}
   */
  sampleCounter = 0;

  /**
   * Creates a new instance of the Neuron class.
   *
   * @param {number} inputCount - The number of inputs to the neuron.
   */
  constructor(inputCount) {
    this.id = `n-${crypto.getRandomValues(new Uint32Array(1))[0].toString(16)}`;
    this.weights = Float64Array.from(
      { length: inputCount },
      () => (Math.random() - 0.5) * 0.5
    );
    this.bias = 0.1;
    this.deltaWeightSums = new Float64Array(inputCount);
  }

  /**
   * Feeds forward the inputs through the neuron and calculates the output.
   * Sigmoid is the activation function used to calculate the output value.
   *
   * @param {Float64Array} inputs - The input values.
   * @returns {number} - The output value of the neuron.
   */
  feedForward(inputs) {
    this.inputs = inputs;
    let weightedSum = 0;
    for (let i = 0; i < inputs.length; i++) {
      weightedSum += inputs[i] * this.weights[i];
    }
    this.output = sigmoid(weightedSum + this.bias);
    return this.output;
  }

  /**
   * Backpropagates the error through the neuron.
   * This accumulates the delta values which are
   * later used to update the weights and bias.
   *
   * @param {number} error - The error value.
   */
  backpropagate(error) {
    this.error = error;
    // remember, this.output is a sigmoid value
    const delta = error * sigmoidDerivative(this.output);
    this.deltaBiasSum += delta;
    for (let i = 0; i < this.inputs.length; i++) {
      this.deltaWeightSums[i] += delta * this.inputs[i];
    }
    this.sampleCounter++;
  }

  /**
   * Updates the weights and bias based on the accumulated deltas
   * and the learning rate.
   *
   * The learning rate determines how much the weights and bias are adjusted
   * during training. A higher learning rate can speed up training, but if it's
   * too high, the network may not converge; overshooting the optimal weights.
   *
   * @param {number} learningRate - The learning rate.
   */
  updateWeights(learningRate) {
    for (let i = 0; i < this.weights.length; i++) {
      const averageDeltaWeight = this.deltaWeightSums[i] / this.sampleCounter;
      this.weights[i] += learningRate * averageDeltaWeight;
    }
    const averageDeltaBias = this.deltaBiasSum / this.sampleCounter;
    this.bias += learningRate * averageDeltaBias;

    // reset the sums and sample counter for the next batch
    this.sampleCounter = 0;
    this.deltaBiasSum = 0;
    this.deltaWeightSums.fill(0);
  }

  /**
   * Calculates the error for a given weight index.
   * It is proportional to the error of the neuron, the weight, and the derivative
   * of the activation function on the output value.
   * 
   * It is used by connected neurons in the previous layer to accumulate their error.
   * This is how the error is backpropagated through the network. It almost seems like
   * magic, but it's just the chain rule of calculus.
   * 
   * @see {@link https://en.wikipedia.org/wiki/Chain_rule}
   *
   * @param {number} index - The index of the weight.
   * @returns {number} - The error value.
   */
  errorForWeight(index) {
    return this.error * this.weights[index] * sigmoidDerivative(this.output);
  }
}

/**
 * Sigmoid is the activation function used by the neurons.
 * Sigmoid produces an S-shaped curve between 0 and 1.
 * Negative inputs produce values close to 0, and
 * positive inputs produce values close to 1.
 *
 * Other activation functions like ReLU (Rectified Linear Unit)
 * may also be used, but sigmoid is simple and works well for
 * this example.
 * 
 * @see {@link https://en.wikipedia.org/wiki/Sigmoid_function}
 *
 * @param {number} x - The input value.
 * @returns {number} - The sigmoid value.
 */
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Calculates the derivative of the sigmoid function for a given value.
 * This gives the slope of the curve at that point. A high slope leads to
 * the weights being adjusted more during training.
 *
 * @param {number} sigmoid - A sigmoid value (0-1) computed earlier.
 * @returns {number} - The sigmoid derivative value.
 */
function sigmoidDerivative(sigmoid) {
  return sigmoid * (1 - sigmoid);
}

/**
 * Random noise data that is labeled as outside the normal range of output.
 * This can be used to train the neural network to recognize when the input
 * data is not a valid element in the expected set.
 * 
 * @param {number} inputLength - The length of the input data.
 * @param {number} outputLength - The length of the output data.
 * @param {number} length - The number of samples to generate.
 * @returns {{ input: Float64Array, output: number[] }[]} An array of objects containing the input and output data.
 */
function noise(inputLength = 28 * 28, outputLength = 11, length = 6000) {
  const output = Array(outputLength).fill(0);
  output[output.length - 1] = 1;
  return Array.from({ length }, () => {
    return {
      input: Float64Array.from({ length: inputLength }, () => Math.random()),
      output,
    };
  });
}

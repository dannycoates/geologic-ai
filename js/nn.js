/**
 * Calculates the sigmoid function of a given value.
 * This is the activation function used by the neurons.
 * Sigmoid produces an S-shaped curve between 0 and 1.
 * Negative inputs produce values close to 0, and 
 * positive inputs produce values close to 1.
 * @param {number} x - The input value.
 * @returns {number} - The sigmoid value.
 */
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Calculates the derivative of the sigmoid function for a given value.
 * @param {number} sigmoid - The input value.
 * @returns {number} - The sigmoid derivative value.
 */
function sigmoidDerivative(sigmoid) {
  return sigmoid * (1 - sigmoid);
}

/**
 * A single neuron in a neural network.
 */
class Neuron {
  /**
   * The inputs to the neuron.
   * Each value is a number between 0 and 1.
   * @type {number[]}
   */
  inputs = [];

  /**
   * The output value of the neuron.
   * It's value is between 0 and 1.
   * @type {number}
   */
  output = 0;

  /**
   * The weights used to calculate the output value.
   * There is one weight per input value.
   * These are updated during training to improve the accuracy of the network.
   * A larger weight value makes it's corresponding input value more influential
   * in the output calculation.
   * @type {number[]}
   */
  weights = [];

  /**
   * The bias of the neuron.
   * It is added to the weighted sum of the inputs to influence the output value
   * and is updated during training. A larger bias value makes it easier for the
   * neuron to fire and produce an output value closer to 1.
   * @type {number}
   */
  bias = 0;

  /**
   * The delta is used to update the weights during training. It represents a
   * magnitude and direction (+/-) that the weights should be changed to reduce
   * the error. It is part of the gradient descent algorithm.
   * It is calculated during backpropagation from the error and the output 
   * derivative. A new delta is calculated during backpropagation and used to 
   * update the weights.
   * @type {number}
   */
  delta = 0;

  /**
   * Creates a new instance of the Neuron class.
   * @param {number} inputCount - The number of inputs to the neuron.
   */
  constructor(inputCount) {
    this.weights = Array.from(
      { length: inputCount },
      () => Math.random() * 2 - 1 // random value between -1 and 1
    );
    this.bias = 0.1;
  }

  /**
   * Feeds forward the inputs through the neuron and calculates the output.
   * Sigmoid is the activation function used to calculate the output value.
   * @param {number[]} inputs - The input values.
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
   * This sets the delta value for the neuron which is used to update the weights
   * of this neuron and the error of connected neurons in the previous layer.
   * @param {number} error - The error value.
   */
  backpropagate(error) {
    this.delta = error * sigmoidDerivative(this.output);
  }

  /**
   * Updates the weights and bias of the neuron.
   * @param {number} learningRate - The learning rate.
   */
  updateWeights(learningRate) {
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] += learningRate * this.delta * this.inputs[i];
    }
    this.bias += learningRate * this.delta;
  }

  /**
   * Calculates the error for a given weight index.
   * This is used by connected neurons in the previous layer.
   * @param {number} index - The index of the weight.
   * @returns {number} - The error value.
   */
  errorForWeight(index) {
    return this.delta * this.weights[index];
  }
}

/**
 * A layer of neurons in a neural network.
 */
class Layer {
  /**
   * The neurons in the layer.
   * @type {Neuron[]}
   */
  neurons = [];

  /**
   * Creates a new instance of the Layer class.
   * @param {number} neuronCount - The number of neurons in the layer.
   * @param {number} inputCount - The number of inputs per neuron.
   */
  constructor(neuronCount, inputCount) {
    this.neurons = Array.from(
      { length: neuronCount },
      () => new Neuron(inputCount)
    );
  }

  get size() {
    return this.neurons.length;
  }

  /**
   * Feeds forward the inputs through the layer and calculates the outputs.
   * @param {number[]} inputs - The input values.
   * @returns {number[]} - The output values of the layer.
   */
  feedForward(inputs) {
    return this.neurons.map((neuron) => neuron.feedForward(inputs));
  }

  /**
   * Backpropagates the error through the layer.
   * @param {Layer} nextLayer - The next layer in the neural network.
   */
  backpropagate(nextLayer) {
    for (let i = 0; i < this.neurons.length; i++) {
      let error = 0;
      for (const neuron of nextLayer.neurons) {
        error += neuron.errorForWeight(i);
      }
      this.neurons[i].backpropagate(error);
    }
  }

  /**
   * Updates the weights and biases of the neurons in the layer.
   * @param {number} learningRate - The learning rate.
   */
  updateWeights(learningRate) {
    for (const neuron of this.neurons) {
      neuron.updateWeights(learningRate);
    }
  }
}

/**
 * A fully connected neural network.
 */
export class NeuralNetwork {
  /**
   * The layers of the neural network.
   * @type {Layer[]}
   */
  layers = [];

  /**
   * Creates a new instance of the NeuralNetwork class.
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
   * @param {number[]} inputs - The input values.
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
   * @param {number[]} targetOutputs - The target output values from the training data.
   * @returns {number} - The mean squared error of this iteration.
   */
  backpropagate(targetOutputs) {
    let sumSquaredError = 0;
    // We process the layers in reverse order. The ouput layer
    // uses the target outputs, while the hidden layers use the
    // weights and deltas from the next layer.
    for (let i = 0; i < targetOutputs.length; i++) {
      const neuron = this.outputLayer.neurons[i];
      const target = targetOutputs[i];
      const error = target - neuron.output;
      sumSquaredError += error * error;
      neuron.backpropagate(error);
    }

    for (let i = this.layers.length - 2; i >= 0; i--) {
      this.layers[i].backpropagate(this.layers[i + 1]);
    }
    return sumSquaredError / targetOutputs.length;
  }

  /**
   * Updates the weights and biases of the neural network.
   * @param {number} learningRate - The learning rate.
   */
  updateWeights(learningRate) {
    for (const layer of this.layers) {
      layer.updateWeights(learningRate);
    }
  }

  /**
   * Trains the neural network using the given inputs and target outputs.
   * @param {import("./mnist").LabelledInputData[]} inputs - The input values.
   * @param {number} learningRate - The learning rate.
   * @param {number} epochs - The number of rounds of training to do over all the inputs.
   * @param {number} batchSize - The size of each training batch.
   */
  train(inputs, learningRate, epochs, batchSize) {
    const trainingData = inputs.map(({ input, label }) => ({
      input,
      // Transform the label into an array representing the desired output layer values.
      output: Array.from({ length: this.outputLayer.size }, (_, i) =>
        i === label ? 1 : 0
      ),
    }));
    for (let e = 0; e < epochs; e++) {
      console.time(`Epoch ${e + 1}/${epochs}`);
      const epochData = trainingData
        .concat(
          noise(
            trainingData[0].input.length,
            trainingData[0].output.length,
            trainingData.length / 10
          )
        )
        .sort(() => Math.random() - 0.5);
      let error = 0;
      for (let i = 0; i < epochData.length; i += batchSize) {
        const batchData = epochData.slice(i, i + batchSize);
        for (let j = 0; j < batchData.length; j++) {
          this.feedForward(batchData[j].input);
          error += this.backpropagate(batchData[j].output);
          this.updateWeights(learningRate);
        }
      }
      console.timeEnd(`Epoch ${e + 1}/${epochs}`);
      console.log(`mean squared error: ${error / epochData.length}`);
    }
  }
}

/**
 * Random noise data that is labeled as outside the normal range.
 * @param {number} inputLength - The length of the input data.
 * @param {number} outputLength - The length of the output data.
 * @param {number} length - The number of samples to generate.
 * @returns {{ input: number[], output: number[] }[]} An array of objects containing the input and output data.
 */
function noise(inputLength = 28 * 28, outputLength = 11, length = 6000) {
  const output = new Array(outputLength).fill(0);
  output[output.length - 1] = 1;
  return Array.from({ length }, () => {
    return {
      input: Array.from({ length: inputLength }, () => Math.random()),
      output,
    };
  });
}

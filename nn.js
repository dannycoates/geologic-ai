/**
 * Calculates the sigmoid function of a given value.
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
 * Represents a single neuron in a neural network.
 */
class Neuron {
  /**
   * The inputs to the neuron.
   * @type {number[]}
   */
  inputs = [];

  /**
   * The output value of the neuron.
   * @type {number}
   */
  output = 0;

  /**
   * The weights of the neuron's inputs.
   * @type {number[]}
   */
  weights = [];

  /**
   * The bias of the neuron.
   * @type {number}
   */
  bias = 0;

  /**
   * The delta value used in backpropagation.
   * @type {number}
   */
  delta = 0;

  /**
   * Creates a new instance of the Neuron class.
   * @param {number} inputCount - The number of inputs to the neuron.
   */
  constructor(inputCount) {
    this.weights = Array.from({ length: inputCount }, () => Math.random() - 0.5);
    this.bias = 0.5;
  }

  /**
   * Feeds forward the inputs through the neuron and calculates the output.
   * @param {number[]} inputs - The input values.
   * @returns {number} - The output value of the neuron.
   */
  feedForward(inputs) {
    this.inputs = inputs;
    this.output = sigmoid(
      inputs.reduce((sum, input, i) => sum + input * this.weights[i], this.bias)
    );
    return this.output;
  }

  /**
   * Backpropagates the error through the neuron.
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
}

/**
 * Represents a layer of neurons in a neural network.
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
      const neuron = this.neurons[i];
      let error = nextLayer.neurons.reduce(
        (sum, nextNeuron) => sum + nextNeuron.weights[i] * nextNeuron.delta,
        0
      );
      neuron.backpropagate(error);
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
 * Represents a neural network.
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
   */
  constructor(...layerSizes) {
    for (let i = 1; i < layerSizes.length; i++) {
      this.layers.push(new Layer(layerSizes[i], layerSizes[i - 1]));
    }
  }

  /**
   * Feeds forward the inputs through the neural network and calculates the outputs.
   * @param {number[]} inputs - The input values.
   * @returns {number[]} - The output values of the neural network.
   */
  feedForward(inputs) {
    for (const layer of this.layers) {
      inputs = layer.feedForward(inputs);
    }
    return inputs;
  }

  /**
   * Backpropagates the error through the neural network.
   * @param {number[]} targetOutputs - The target output values from the training data.
   */
  backpropagate(targetOutputs) {
    // We process the layers in reverse order. The ouput layer
    // uses the target outputs, while the hidden layers use the
    // weights and deltas from the next layer.
    const outputLayer = this.layers[this.layers.length - 1];
    for (let i = 0; i < targetOutputs.length; i++) {
      const neuron = outputLayer.neurons[i];
      const target = targetOutputs[i];
      neuron.backpropagate(target - neuron.output);
    }

    for (let i = this.layers.length - 2; i >= 0; i--) {
      this.layers[i].backpropagate(this.layers[i + 1]);
    }
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
   * @param {{ image: number[], label: number }[]} inputs - The input values.
   * @param {number} learningRate - The learning rate.
   * @param {number} epochs - The number of training epochs.
   * @param {number} batchSize - The size of each training batch.
   */
  train(inputs, learningRate, epochs, batchSize) {
    const trainingData = inputs.map(({ image, label }) => ({
      image,
      output: Array.from({ length: 10 }, (_, i) => (i === label ? 1 : 0)),
    }));
    for (let e = 0; e < epochs; e++) {
      console.log(`Epoch ${e + 1}/${epochs}`);
      const epochData = trainingData.sort(() => Math.random() - 0.5);
      for (let i = 0; i < epochData.length; i += batchSize) {
        const batchData = epochData.slice(i, i + batchSize);

        for (let j = 0; j < batchData.length; j++) {
          this.feedForward(batchData[j].image);
          this.backpropagate(batchData[j].output);
          this.updateWeights(learningRate);
        }
      }
    }
  }
}

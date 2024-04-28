import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js";

export class TFNetwork {

  static async create(...layerSizes) {
    // await tf.setBackend("wasm");
    return new TFNetwork(...layerSizes);
  }

  constructor(...layerSizes) {
    const inputCount = layerSizes.shift();
    this.inputSizes = [inputCount, ...layerSizes];
    this.model = tf.sequential();
    this.model.add(
      tf.layers.dense({ name: 'input', units: inputCount, inputDim: inputCount, useBias: true })
    );
    for (let units of layerSizes) {
      this.model.add(tf.layers.dense({ units, activation: "sigmoid", useBias: true}));
    }
    this.model.compile({ loss: "meanSquaredError", optimizer: tf.train.sgd(3.0) });
    this.meanSquaredError = 0;
  }

  convertData(data) {
    const inputSize = this.inputSizes[0];
    const outputSize = this.inputSizes[this.inputSizes.length - 1];
    const trainingData = new Float32Array(data.length * inputSize);
    const labels = new Float32Array(data.length * outputSize);
    for (let i = 0; i < data.length; i++) {
      trainingData.set(data[i].input, i * inputSize);
      labels.set(
        Float32Array.from({ length: outputSize }, (_, j) =>
          j === data[i].label ? 1 : 0
        ),
        i * outputSize
      );
    }
    const inputs = tf.tensor2d(trainingData, [data.length, inputSize]);
    const outputs = tf.tensor2d(labels, [data.length, outputSize]);
    return {
      inputs,
      outputs,
    }
  }

  /**
   * Trains the model.
   * @param {{inputs: any, outputs: any }} data - The input data.
   * @params {number} learningRate - The learning rate.
   * @params {number} batchSize - The batch size.
   * @returns {Promise<void>}
   */
  async train(data, learningRate = 1, batchSize = 1) {
    await this.model.fit(data.inputs, data.outputs, {
      batchSize,
      epochs: 1,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          this.meanSquaredError = logs.loss;
        },
      },
      yieldEvery: "never",
    });
  }

  feedForward(inputs) {
    const x = tf.tensor2d(inputs, [1, this.inputSizes[0]]);
    return this.model.predict(x).dataSync();
  }

  uiData() {
    let id = 0;
    const layers = [];
    for (let i = 1; i < this.model.layers.length; i++) {
      const weightCount = this.inputSizes[i -1];
      const allWeights = this.model.layers[i].getWeights()[0].dataSync();
      const neurons = [];
      for (let j = 0; j < allWeights.length; j += weightCount) {
        const weights = allWeights.slice(j, j + weightCount);
        neurons.push({ id: `n_${id++}`, weights });
      }
      layers.push({ id: `l_${id++}`, neurons });
    }
    return {
      meanSquaredError: this.meanSquaredError,
      layers,
    };
  }
}

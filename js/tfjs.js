import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js";

export class TFNetwork {
  inputSizes = [];
  learningRate = 1;
  loss = 0;
  model = null;
  trainingData = null;

  static async create(settings, ...layerSizes) {
    // await tf.setBackend("wasm");
    return new TFNetwork(settings, ...layerSizes);
  }

  constructor(settings, ...layerSizes) {
    this.inputSizes = layerSizes.slice();
    const inputCount = layerSizes.shift();
    this.learningRate = settings.learningRate ?? 1;
    this.model = tf.sequential();
    this.model.add(
      tf.layers.dense({
        name: "input",
        units: layerSizes.shift(),
        activation: "sigmoid",
        inputDim: inputCount,
        useBias: true,
        kernelInitializer: "randomUniform",
      })
    );
    for (let units of layerSizes) {
      this.model.add(
        tf.layers.dense({ units, activation: "sigmoid", useBias: true })
      );
    }
    this.model.compile({
      loss: "meanSquaredError",
      // optimizer: "sgd"
      optimizer: tf.train.sgd(this.learningRate),
    });
    this.model.summary();
  }

  setTrainingData(data) {
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
    this.trainingData = { inputs, outputs };
  }

  /**
   * Trains the model.
   * @params {number} batchSize - The batch size.
   * @returns {Promise<void>}
   */
  async train(batchSize = 1) {
    const { inputs, outputs } = this.trainingData;
    await this.model.fit(inputs, outputs, {
      batchSize,
      epochs: 1,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          this.loss = logs.loss;
        },
      },
      yieldEvery: "never",
    });
  }

  feedForward(inputs) {
    const x = tf.tensor2d(Float32Array.from(inputs), [1, this.inputSizes[0]]);
    return this.model.predict(x).dataSync();
  }

  uiData() {
    let id = 0;
    const layers = [];
    for (let i = 0; i < this.model.layers.length; i++) {
      const weightCount = this.inputSizes[i];
      // weights[0] is the input weights, [1] is the biases
      const allWeights = this.model.layers[i].getWeights(true)[0].dataSync();
      const neurons = [];
      for (let j = 0; j < allWeights.length; j += weightCount) {
        const weights = allWeights.subarray(j, j + weightCount);
        neurons.push({ id: `n_${id++}`, weights });
      }
      layers.push({ id: `l_${id++}`, neurons });
    }
    return {
      loss: this.loss,
      layers,
    };
  }
}

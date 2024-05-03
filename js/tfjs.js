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

  static async load() {
    const model = await tf.loadLayersModel("indexeddb://geologic-model");
    const network = new TFNetwork(model);
    return network;
  }

  constructor(settingsOrModel, ...layerSizes) {
    if (settingsOrModel instanceof tf.Sequential) {
      this.model = settingsOrModel;
      this.inputSizes = [this.model.inputs[0].shape[1]].concat(this.model.layers.map((l) => l.units));
      return;
    }
    this.inputSizes = layerSizes.slice();
    const inputCount = layerSizes.shift();
    this.learningRate = settingsOrModel.learningRate ?? 1;
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

  async save() {
    return await this.model.save("indexeddb://geologic-model");
  }

  uiData() {
    const layerCount = this.model.layers.length;
    let id = 0;
    const layers = Array(layerCount);
    for (let i = 0; i < layerCount; i++) {
      const weightCount = this.inputSizes[i];
      // weights[0] is the input weights, [1] is the biases
      const allWeights = this.model.layers[i].getWeights()[0].arraySync();
      const neuronCount = allWeights[0].length
      const neurons = Array(neuronCount);
      for (let n = 0; n < neuronCount; n++) {
        const weights = Array(weightCount);
        for (let j = 0; j < weightCount; j++) {
          weights[j] = allWeights[j][n];
        }
        neurons[n] = { id: `n_${id++}`, weights };
      }
      layers[i] = { id: `l_${id++}`, neurons };
    }
    return {
      loss: this.loss,
      layers,
    };
  }
}

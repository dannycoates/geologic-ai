import { NeuralNetwork } from "./nn.js";
import { trainingData, testData } from "./mnist.js";

/**
 * The neural network
 * @type {NeuralNetwork}
 */
let nn;

/**
 * The training data
 * @type {Array<{input: Float64Array, label: number}>}
 */
let training;

/**
 * The learning rate
 * @type {number}
 */
let learningRate = 3;

/**
 * The batch size
 * @type {number}
 */
let batchSize = 100;

/**
 * The worker message event listener
 * @param {MessageEvent} event
 * @returns {void}
 * @listens MessageEvent
 */
onmessage = async function (event) {
  const { data } = event;
  const { type, payload } = data;
  switch (type) {
    case "init": {
      training = await trainingData();
      nn = new NeuralNetwork(784, ...payload.sizes);
      learningRate = payload.learningRate;
      batchSize = payload.batchSize;
      postMessage({ type: "init", payload: this.nn });
      break;
    }
    case "train": {
      nn.train(training, learningRate, batchSize);
      postMessage({ type: "progress", payload: nn });
      break;
    }
    case "test": {
      const test = await testData();
      let correct = 0;
      // We save one example of each incorrectly predicted digit
      let incorrect = new Map();
      for (let i = 0; i < test.length; i++) {
        const outputs = nn.feedForward(test[i].input);
        const label = outputs.indexOf(Math.max(...outputs));
        const expected = test[i].label;
        if (label === expected) {
          correct++;
        } else {
          incorrect.set(expected, test[i].input);
        }
      }
      postMessage({
        type: "test",
        payload: {
          correct: correct / test.length,
          incorrect,
        },
      });
    }
    default:
      break;
  }
};

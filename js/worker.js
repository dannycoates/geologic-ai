import { NeuralNetwork } from "./geologic.js";
import { trainingData, testData } from "./mnist.js";
import { TFNetwork } from "./tfjs.js";

/**
 * The neural network
 * @type {NeuralNetwork}
 */
let nn;

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
      const NN = payload.engine === "geologic" ? NeuralNetwork : TFNetwork;
      nn = await NN.create(
        { learningRate: payload.learningRate },
        784,
        ...payload.sizes
      );
      nn.setTrainingData(await trainingData());
      batchSize = payload.batchSize;
      postMessage({ type: "init", payload: nn.uiData() });
      break;
    }
    case "train": {
      await nn.train(batchSize);
      postMessage({ type: "progress", payload: nn.uiData() });
      break;
    }
    case "load": {
      const NN = payload.engine === "geologic" ? NeuralNetwork : TFNetwork;
      nn = await NN.load();
      postMessage({ type: "loaded", payload: nn.uiData() });
      break;
    }
    case "test": {
      await nn.save()
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

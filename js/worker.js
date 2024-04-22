import { NeuralNetwork } from "./nn.js";
import { trainingData, testData } from "./mnist.js";

onmessage = async function (event) {
  const { data } = event;
  const { type, payload } = data;
  switch (type) {
    case "init": {
      const training = await trainingData();
      this.nn = new NeuralNetwork(784, ...payload.sizes);
      this.gen = this.nn.train(
        training,
        payload.learningRate,
        payload.epochs,
        payload.batchSize
      );
      postMessage({ type: "init", payload: this.nn });
      break;
    }
    case "gen": {
      const { value, done } = this.gen.next();
      if (done) {
        postMessage({ type: "done", payload: this.nn });
      } else {
        postMessage({ type: "progress", payload: value });
      }
      break;
    }
    case "test": {
      const test = await testData();
      let correct = 0;
      for (let i = 0; i < test.length; i++) {
        const outputs = this.nn.feedForward(test[i].input);
        const label = outputs.indexOf(Math.max(...outputs));
        const expected = test[i].label;
        if (label === expected) correct++;
      }
      postMessage({ type: "test", payload: correct / test.length });
    }
    default:
      break;
  }
};

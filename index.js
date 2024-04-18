import { trainingData, testData } from "./mnist.js";
import { NeuralNetwork } from "./nn.js";

/**
 * The entry point of the program.
 */
async function main() {
  const data = await trainingData();
  const network = new NeuralNetwork(784, 16, 16, 10);
  network.train(data, 0.1, 10, 1000);

  const test = await testData();
  let correct = 0;
  for (let i = 0; i < test.length; i++) {
    const outputs = network.feedForward(test[i].image);
    const label = outputs.indexOf(Math.max(...outputs));
    const expected = test[i].label;
    if (label === expected) correct++;
  }
  console.log(correct)
}

main();
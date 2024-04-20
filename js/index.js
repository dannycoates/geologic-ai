import { trainingData, testData } from "./mnist.js";
import { NeuralNetwork } from "./nn.js";

/**
 * The entry point of the program.
 */
async function main() {
  const data = await trainingData();
  const first = data[0];
  // The output layer has 11 neurons, one for each digit 0-9 and one for no digit.
  const network = new NeuralNetwork(first.input.length, 49, 11);
  network.train(data, 3, 50, 100);

  const test = await testData();
  let correct = 0;
  for (let i = 0; i < test.length; i++) {
    const outputs = network.feedForward(test[i].input);
    const label = outputs.indexOf(Math.max(...outputs));
    const expected = test[i].label;
    if (label === expected) correct++;
  }
  console.log(correct)
}

main();
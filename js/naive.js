export class NaiveDigitPredictor {
  /**
   * Creates a new NaiveDigitPredictor
   *
   * @param {Array<{label: number, input: Float32Array}>} trainingData
   */
  constructor(trainingData) {
    const inputSize = trainingData[0].input.length;
    this.templates = new Map(
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((digit) => [
        digit,
        new Float32Array(inputSize),
      ])
    );
    for (const { label, input } of trainingData) {
      const template = this.templates.get(label);
      for (let i = 0; i < inputSize; i++) {
        template[i] += input[i] / trainingData.length;
      }
    }
  }

  /**
   * Predicts the digit of the input
   *
   * @param {Float32Array} input
   * @returns {number}
   */
  predict(input) {
    console.assert(
      input.length === this.templates.get(0).length,
      "Input size must match the template size."
    );
    let min = Infinity;
    let prediction = NaN;
    for (const [digit, template] of this.templates) {
      let difference = 0;
      for (let i = 0; i < template.length; i++) {
        difference += Math.abs(template[i] - input[i]);
      }
      if (difference < min) {
        min = difference;
        prediction = digit;
      }
    }
    return prediction;
  }
}

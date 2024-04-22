
async function fetchFile(path) {
  const response = await fetch(new URL(path, import.meta.url));
  return await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}

/**
 * Parses the header of the MNIST data file.
 * @param {ArrayBuffer} buffer - The buffer containing the header data.
 * @returns {{ magic: number, count: number }} An object containing the magic number and count.
 */
function parseHeader(buffer) {
  const view = new DataView(buffer);
  const magic = view.getUint32(0);
  const count = view.getUint32(4);
  return { magic, count };
}

/**
 * Parses the image data from the MNIST data file.
 * @param {ArrayBuffer} buffer - The buffer containing the image data.
 * @param {number} count - The number of images to parse.
 * @returns {Float64Array[]} An array of image data, where each image is represented as an array of pixel values.
 */
function parseImages(buffer, count) {
  const images = new Array(count);
  const view = new DataView(buffer);
  let index = 8;
  const rows = view.getUint32(index); //buffer.readUInt32BE(index);
  index += 4;
  const cols = view.getUint32(index) //buffer.readUInt32BE(index);
  index += 4;
  for (let i = 0; i < count; i++) {
    images[i] = Float64Array.from({ length: rows * cols }, () => view.getUint8(index++) / 255);
  }
  return images;
}

/**
 * Parses the label data from the MNIST data file.
 * @param {ArrayBuffer} buffer - The buffer containing the label data.
 * @returns {Uint8Array} An array of label data.
 */
function parseLabels(buffer) {
  return new Uint8Array(buffer.slice(8));
}

/**
 * Parses the buffer data from the MNIST data file.
 * @param {ArrayBuffer} buffer - The buffer containing the data.
 * @returns {Float64Array[]|Uint8Array|null} Parsed data based on the magic number in the header.
 */
function parseBuffer(buffer) {
  const { magic, count } = parseHeader(buffer);
  if (magic === 2051) return parseImages(buffer, count);
  if (magic === 2049) return parseLabels(buffer);
  return null;
}

/**
 * @typedef {{ input: Float64Array, label: number }} LabelledInputData
 */

/**
 * Retrieves the data from the MNIST dataset.
 * @param {string} imagePath - The path to the image data file.
 * @param {string} labelPath - The path to the label data file.
 * @returns {Promise<LabelledInputData[]>} A promise that resolves to an array of objects containing the input and label data.
 */
async function getData(imagePath, labelPath) {
  const rawImages = await fetchFile(imagePath);
  const rawLabels = await fetchFile(labelPath);
  const images = parseBuffer(rawImages);
  const labels = parseBuffer(rawLabels)
  return images.map((input, i) => ({ input, label: labels[i] }));
}

/**
 * Retrieves the training data from the MNIST dataset.
 * @returns {Promise<LabelledInputData[]>>} A promise that resolves to an array of objects containing the input and label data.
 */
export function trainingData() {
  return getData("../data/train-images-idx3-ubyte.gz", "../data/train-labels-idx1-ubyte.gz");
}

/**
 * Retrieves the test data from the MNIST dataset.
 * @returns {Promise<LabelledInputData[]>>} A promise that resolves to an array of objects containing the input and label data.
 */
export function testData() {
  return getData("../data/t10k-images-idx3-ubyte.gz", "../data/t10k-labels-idx1-ubyte.gz");
}

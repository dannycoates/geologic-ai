import { readFile } from "node:fs/promises";
import { gunzip } from "node:zlib";
import { promisify } from "node:util";

const gunzipAsync = promisify(gunzip);

/**
 * Parses the header of the MNIST data file.
 * @param {Buffer} buffer - The buffer containing the header data.
 * @returns {{ magic: number, count: number }} An object containing the magic number and count.
 */
function parseHeader(buffer) {
  const magic = buffer.readUInt32BE(0);
  const count = buffer.readUInt32BE(4);
  return { magic, count };
}

/**
 * Parses the image data from the MNIST data file.
 * @param {Buffer} buffer - The buffer containing the image data.
 * @param {number} count - The number of images to parse.
 * @returns {number[][]} An array of image data, where each image is represented as an array of pixel values.
 */
function parseImages(buffer, count) {
  const images = new Array(count);
  let index = 0;
  const rows = buffer.readUInt32BE(index);
  index += 4;
  const cols = buffer.readUInt32BE(index);
  index += 4;
  for (let i = 0; i < count; i++) {
    images[i] = Array.from({ length: rows * cols }, () => buffer[index++] / 255);
  }
  return images;
}

/**
 * Parses the label data from the MNIST data file.
 * @param {Buffer} buffer - The buffer containing the label data.
 * @returns {number[]} An array of label data.
 */
function parseLabels(buffer) {
  return Array.from(buffer);
}

/**
 * Parses the buffer data from the MNIST data file.
 * @param {Buffer} buffer - The buffer containing the data.
 * @returns {number[][]|number[]|null} Parsed data based on the magic number in the header.
 */
function parseBuffer(buffer) {
  const { magic, count } = parseHeader(buffer);
  const data = buffer.subarray(8);
  if (magic === 2051) return parseImages(data, count);
  if (magic === 2049) return parseLabels(data);
  return null;
}

/**
 * @typedef {{ input: number[], label: number }} LabelledInputData
 */

/**
 * Retrieves the data from the MNIST dataset.
 * @param {string} imagePath - The path to the image data file.
 * @param {string} labelPath - The path to the label data file.
 * @returns {Promise<LabelledInputData[]>} A promise that resolves to an array of objects containing the input and label data.
 */
async function getData(imagePath, labelPath) {
  const rawImages = await gunzipAsync(await readFile(imagePath));
  const rawLabels = await gunzipAsync(await readFile(labelPath));
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

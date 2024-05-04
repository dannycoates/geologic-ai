import { NeuralNetwork } from "./geologic.js";
import { TFNetwork } from "./tfjs.js";

/**
 * A class to persist the neural network to IndexedDB.
 * 
 * This is compatible with TensorFlow.js models.
 */
export class PersistentStorage {
  static async saveGeologic(network) {
    function serializeModel(model) {
      const modelTopology = {
        backend: "tensor_flow.js",
        class_name: "Sequential",
        config: {
          layers: model.layers.map((l, i) => ({
            class_name: "Dense",
            config: {
              name: `dense_Dense${i}`,
              units: l.size,
              dtype: "float32",
              activation: "sigmoid",
              use_bias: true,
              trainable: true,
              kernel_initializer: { class_name: "RandomUniform", config: { minval: -0.5, maxval: 0.5 } },
            }
          })),
          name: "sequential_1",
        }
      };
      const weightSpecs = model.layers.flatMap((l, i) => ([
        { dtype: "float32", name: `dense_Dense${i}/kernel`, shape: [l.neurons[0].weights.length, l.size] },
        { dtype: "float32", name: `dense_Dense${i}/bias`, shape: [l.size] }
      ]));
      modelTopology.config.layers[0].config.batch_input_shape = [null, weightSpecs[0].shape[0]];
      const weightData = new Float32Array(weightSpecs.reduce((acc, spec) => acc + spec.shape.reduce((a, b) => a * b), 0));
      let start = 0;
      for (let i = 0; i < model.layers.length; i++) {
        const layer = model.layers[i];
        const weightCount = layer.neurons[0].weights.length;
        for (let w = 0; w < weightCount; w++) {
          for (let n = 0; n < layer.size; n++) {
            weightData[start++] = layer.neurons[n].weights[w];
          }
        }
        for (let j = 0; j < layer.size; j++) {
          weightData[start++] = layer.neurons[j].bias;
        }
      }
      return {
        modelArtifacts: {
        format: "layers-model",
        modelTopology, 
        weightSpecs,
        weightData,
       },
       modelPath: "geologic-model"
      };
    }
    return new Promise((resolve, reject) => {
      const dbopen = self.indexedDB.open("tensorflowjs");
      dbopen.onsuccess = () => {
        const db = dbopen.result;
        const transaction = db.transaction("models_store", "readwrite");
        const store = transaction.objectStore("models_store");
        const request = store.put(serializeModel(network));
        request.onsuccess = () => {
          resolve();
        };
        request.onerror = () => {
          reject(request.error);
        };
      }
      dbopen.onerror = () => {
        reject(dbopen.error);
      };
    });
  }

  static async loadGeologic() {
    function parseModel(model) {
      const layers = model.modelTopology.config.layers;
      const layerSizes = layers[0].config.batch_input_shape.slice(1).concat(layers.map((l) => l.config.units));
      const allWeights = new Float32Array(model.weightData);
      const network = new NeuralNetwork({ learningRate: 3 }, ...layerSizes);
      let start = 0;
      for (let i = 0; i < network.layers.length; i++) {
        const layer = network.layers[i];
        for (let j = 0; j < layerSizes[i]; j++) {
          for (let k = 0; k < layer.neurons.length; k++) {
            layer.neurons[k].weights[j] = allWeights[start++]
          }
        }
        for (let j = 0; j < layer.neurons.length; j++) {
          layer.neurons[j].bias = allWeights[start++]
        }
      }
      return network;
    }
    return new Promise((resolve, reject) => {
      const dbopen = self.indexedDB.open("tensorflowjs");
      dbopen.onsuccess = () => {
        const db = dbopen.result;
        const transaction = db.transaction("models_store", "readonly");
        const store = transaction.objectStore("models_store");
        const request = store.get("geologic-model");
        request.onsuccess = () => {
          resolve(parseModel(request.result.modelArtifacts));
        };
        request.onerror = () => {
          reject(request.error);
        };
      }
      dbopen.onerror = () => {
        reject(dbopen.error);
      };
    });
  }

  static async saveTF(network) {
    return network.model.save("indexeddb://geologic-model");
  }

  static async loadTF() {
    const model = await tf.loadLayersModel("indexeddb://geologic-model");
    const network = new TFNetwork(model);
    return network;
  }

  static save(network) {
    return network instanceof NeuralNetwork ? PersistentStorage.saveGeologic(network) : PersistentStorage.saveTF(network);
  }

  static load(engine) {
    return engine === "geologic" ? PersistentStorage.loadGeologic() : PersistentStorage.loadTF();
  }
}
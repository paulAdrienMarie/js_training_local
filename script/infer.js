import * as ort from "/dist/ort.training.wasm.min.js";

let inferenceSession = null;
let pre = null;

export async function loadJSON(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const config = await response.json();
    return config;
  } catch (error) {
    console.error("Error loading config", error);
  }
}

async function loadInferenceSession() {
  let MODEL_PATH = "/model/inference.onnx";
  console.log("Loading Inference Session");

  try {
    inferenceSession = await ort.InferenceSession.create(MODEL_PATH);
    console.log("Inference Session successfully loaded");
  } catch (err) {
    console.log("Error loading the Inference Session:", err);
    throw err;
  }
}

export async function toTensorAndResize(base64Data) {
  pre = await loadJSON("script/preprocessor_config.json");

  const input_size = pre.size.width;
  const shape = [1, 3, input_size, input_size];

  const img = new Image();
  img.src = base64Data;

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  canvas.width = img.width;
  canvas.height = img.height;

  ctx.drawImage(img, 0, 0, img.width, img.height);

  const imageData = ctx.getImageData(0, 0, img.width, img.height);

  const resizeCanvas = document.createElement("canvas");
  const resizeCtx = resizeCanvas.getContext("2d");
  resizeCanvas.width = input_size;
  resizeCanvas.height = input_size;

  resizeCtx.drawImage(
    canvas,
    0,
    0,
    img.width,
    img.height,
    0,
    0,
    input_size,
    input_size
  );

  const resizedImageData = resizeCtx.getImageData(0, 0, input_size, input_size);

  const dataFromImage = new Float32Array(3 * input_size * input_size);

  for (let i = 0; i < input_size * input_size; i++) {
    dataFromImage[i] = resizedImageData.data[i * 4]; // R
    dataFromImage[i + input_size * input_size] =
      resizedImageData.data[i * 4 + 1]; // G
    dataFromImage[i + 2 * input_size * input_size] =
      resizedImageData.data[i * 4 + 2]; // B
  }

  const imageTensor = new ort.Tensor("float32", dataFromImage, shape);

  return imageTensor;
}

export async function preprocessImage(tensor) {

  const imageMean = pre.image_mean;
  const imageStd = pre.image_std;

  let data = await tensor.getData();

  data = data.map(function (value) {
    return (value / 255.0 - imageMean[0]) / imageStd[0];
  });

  let normalizedTensor = new ort.Tensor("float32", data, [1, 3, 224, 224]);

  return normalizedTensor;
}

function softmax(arr) {
  return arr.map(function (value, index) {
    return (
      Math.exp(value) /
      arr
        .map(function (y /*value*/) {
          return Math.exp(y);
        })
        .reduce(function (a, b) {
          return a + b;
        })
    );
  });
}

function argsort(array) {
  // Convert Float32Array to a regular array with indices
  const arrayWithIndices = Array.from(array).map((value, index) => ({
    value,
    index,
  }));

  // Sort by value in descending order and map to indices
  arrayWithIndices.sort((a, b) => b.value - a.value);

  // Extract sorted indices
  return arrayWithIndices.map((item) => item.index);
}

export async function predict(base64Data) {
  if (!inferenceSession) {
    console.log("Inference session not loaded yet, waiting...");
    await loadInferenceSession();
  }

  // Preprocess the input_image
  let imageTensor = await toTensorAndResize(base64Data);
  let inputImage = await preprocessImage(imageTensor);

  // Prepare the input of the session
  let feeds = {
    pixel_values: inputImage,
  };

  // Run the session
  let results = await inferenceSession.run(feeds);

  // Retrieve the logits of the outputs
  let logits = results["logits"];

  // Transform the logits into a probability distribution
  let probs = softmax(logits.cpuData);

  // Sort the probs in descending order
  let sorted_indices = argsort(probs);

  const config = await loadJSON("script/config.json");
  const id2label = config.id2label;

  let labels = {};
  sorted_indices.slice(0, 5).forEach((i, x) => {
    labels[id2label[i.toString()]] = probs[i];
  });

  return labels;
}

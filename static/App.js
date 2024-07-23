import * as ort from "/dist/ort.training.wasm.min.js";

ort.env.wasm.wasmPaths = "/dist/";
ort.env.wasm.numThreads = 1;

let inferenceSession = null;

let trainingWorker = new Worker("/script/training-worker.js", {
  type: "module",
});

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

async function loadJSON(url) {
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

const config = await loadJSON("/script/config.json");
const pre = await loadJSON("/script/preprocessor_config.json");

async function toTensorAndResize(base64Data) {
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

async function preprocessImage(tensor) {
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
  const arrayWithIndices = Array.from(array).map((value, index) => ({
    value,
    index,
  }));

  arrayWithIndices.sort((a, b) => b.value - a.value);

  return arrayWithIndices.map((item) => item.index);
}

function createTargetTensor(new_class) {
  const index = config.label2id[new_class];
  const shape = [1, 1000];
  const low_value = -3.5;

  const size = shape.reduce((a, b) => a * b);
  let data = new Float32Array(size).fill(low_value);
  data[index] = -low_value;

  return new ort.Tensor("float32", data, shape);
}

function augmentImage(imageTensor) {
  let augmentedImage = tf.expandDims(imageTensor, 0);

  if (Math.random() > 0.5) {
    augmentedImage = tf.image.flipLeftRight(augmentedImage);
  }

  const rotations = [0, 90, 180, 270];
  const angle = rotations[Math.floor(Math.random() * rotations.length)];
  augmentedImage = tf.image.rotateWithOffset(augmentedImage, angle / 90);

  return augmentedImage;
}

async function preprocessImageTraining(base64Data, pre) {
  const numImages = 7;
  const images = [];
  const inputSize = {
    width: pre.size.width,
    height: pre.size.height,
  };

  let image = await toTensorAndResize(base64Data);
  const imageData_ = await image.getData();

  for (let i = 0; i < numImages; i++) {
    let imageTensor = tf.tensor(imageData_, [3, 224, 224], "float32");

    imageTensor = tf.transpose(imageTensor, [1, 2, 0]);

    let augmentedTensor = augmentImage(imageTensor);

    augmentedTensor = tf.transpose(augmentedTensor, [0, 3, 1, 2]);

    let data_ = await augmentedTensor.data();
    let shape = augmentedTensor.shape;

    augmentedTensor = new ort.Tensor("float32", data_, shape);

    augmentedTensor = await preprocessImage(augmentedTensor);

    images.push(augmentedTensor);
  }

  return images;
}

export async function predict(base64Data) {
  if (!inferenceSession) {
    console.log("Inference session not loaded yet, waiting...");
    await loadInferenceSession();
  }

  let imageTensor = await toTensorAndResize(base64Data);
  let inputImage = await preprocessImage(imageTensor);

  let feeds = {
    pixel_values: inputImage,
  };

  let results = await inferenceSession.run(feeds);
  let logits = results["logits"];

  let probs = softmax(logits.cpuData);
  let sorted_indices = argsort(probs);

  const id2label = config.id2label;

  let labels = {};
  sorted_indices.slice(0, 7).forEach((i, x) => {
    labels[id2label[i.toString()]] = probs[i];
  });

  return labels;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function train(base64Data, new_class) {
  try {
    let i = 0;
    const console = document.getElementById("console");
    console.hidden = false;
    trainingWorker.onmessage = async function (message) {
      let text_area = document.createElement("p");

      if (message.data.loss) {
        text_area.innerHTML = `<strong>LOSS:</strong> ${message.data.loss.toString()}`;
      } else {
        text_area.innerHTML = `<strong>${message.data.epochMessage.toString()}</strong>`;
        if (message.data.reload == true) {
          await sleep(2000);
          location.reload();
        }
      }

      text_area.id = `output_${i}`;
      console.appendChild(text_area);
      i++;
    };

    let images = await preprocessImageTraining(base64Data, pre);

    let serializedImages = images.map((img) => ({
      data: Array.from(img.data),
      dims: img.dims,
      type: img.type,
    }));

    const target_tensor = createTargetTensor(new_class);
    const serializedTarget = {
      data: Array.from(target_tensor.data),
      dims: target_tensor.dims,
      type: target_tensor.type,
    };

    let data = {
      images: serializedImages,
      target_tensor: serializedTarget,
    };

    trainingWorker.postMessage(data);
  } catch (err) {
    console.error("Error during training:", err);
    throw err;
  }
}

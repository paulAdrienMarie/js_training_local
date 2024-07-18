import * as ort from "/dist/ort.training.wasm.min.js";

ort.env.wasm.wasmPaths = "/dist";
ort.env.wasm.numThreads = 1;

const fileUpload = document.getElementById("file-upload");
const imageContainer = document.getElementById("image-container");

fileUpload.addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (!file) {
    return;
  }

  const reader = new FileReader();

  reader.onload = function (e2) {
    imageContainer.innerHTML = "";
    const image = document.createElement("img");
    image.src = e2.target.result;
    image.id = "image-id";
    imageContainer.appendChild(image);
    image.onload = function () {
      detect(image);
    };
  };
  reader.readAsDataURL(file);
});

async function detect(image) {
  const labels = await predict(image.src);
  displayOutput(labels);
}

function displayOutput(data) {
  const labels = data;

  const generated_text = document.createElement("div");
  generated_text.id = "textarea-id";

  const labelsContainer = document.createElement("div");
  labelsContainer.id = "labels-container";

  for (const label in labels) {
    if (labels.hasOwnProperty(label)) {
      const checkboxContainer = document.createElement("div");
      checkboxContainer.classList.add("checkbox-container");

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.id = `checkbox-${label}`;
      checkbox.name = label;
      checkbox.value = labels[label];

      const labelElement = document.createElement("label");
      labelElement.htmlFor = `checkbox-${label}`;
      labelElement.innerHTML = `${label}: <span class="bold">${labels[label]}</span>`;

      checkboxContainer.appendChild(checkbox);
      checkboxContainer.appendChild(labelElement);

      labelsContainer.appendChild(checkboxContainer);
    }
  }

  generated_text.appendChild(labelsContainer);
  imageContainer.appendChild(generated_text);
  displayButtons();
}

function displayButtons() {
  const textarea = document.getElementById("textarea-id");
  let buttons = document.createElement("div");
  buttons.id = "buttons-id";
  textarea.appendChild(buttons);

  let button_validate = document.createElement("button");
  button_validate.className = "check_button";
  button_validate.id = "button-validate-id";
  button_validate.innerText = "Validate";

  let button_retrain = document.createElement("button");
  button_retrain.className = "check_button";
  button_retrain.id = "button-retrain-id";
  button_retrain.innerText = "Retrain";

  buttons.appendChild(button_validate);
  buttons.appendChild(button_retrain);

  add_Event("button-retrain-id");
  add_Event("button-validate-id");
}

function add_Event(id) {
  if (id === "button-retrain-id") {
    document.getElementById(id).addEventListener("click", function () {
      const checkedCheckbox = document.querySelector("#labels-container input[type='checkbox']:checked");
      if (checkedCheckbox) {
        const labelElement = document.querySelector(`label[for="${checkedCheckbox.id}"]`);
        const labelText = labelElement.innerText.split(":")[0];
        launch_training(labelText);
      } else {
        alert("Please select a label before launching the training.");
      }
    });
  } else {
    document.getElementById(id).addEventListener("click", function () {
      const image = document.getElementById("image-id");
      imageContainer.removeChild(image);
      const textarea = document.getElementById("textarea-id");
      imageContainer.removeChild(textarea);
      document.getElementById("result").textContent = "";
    });
  }
}

async function launch_training(new_class) {
  let image = document.getElementById("image-id");
  document.getElementById("result").textContent = "Training started with success";
  await train(image.src, new_class);
}

let inferenceSession = null;

ort.env.wasm.wasmPaths = "/dist/";
ort.env.wasm.numThreads = 1;

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
  const low_value = -3;

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
  const numImages = 10;
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
  sorted_indices.slice(0, 5).forEach((i, x) => {
    labels[id2label[i.toString()]] = probs[i];
  });

  return labels;
}

export async function train(base64Data, new_class) {
  try {
    var trainingWorker = new Worker("/script/training-worker.js",{type:"module"});

    trainingWorker.onmessage = function (message) {
      console.log(message);
    };

    console.log(trainingWorker);

    let images = await preprocessImageTraining(base64Data, pre);

    let serializedImages = images.map(img => ({
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

    console.log("DATA", data);

    trainingWorker.postMessage(data);
  } catch (err) {
    console.error("Error during training:", err);
    throw err;
  }
}

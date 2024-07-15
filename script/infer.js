import * as ort from "/dist/ort.training.wasm.min.js";

async function loadConfig(url) {
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

function loadInferenceSession() {
  let MODEL_PATH = "model/inference.onnx";
  console.log("Loading Inference Session");

  try {
    const session = new ort.InferenceSession(MODEL_PATH);
    console.log("Inference Session successfully loaded");
  } catch (err) {
    console.log("Error loading the Inference Session:", err);
    throw err;
  }
}

export async function preprocessImage(base64Data) {
  // Load configuration asynchronously
  const pre = await loadConfig("script/preprocessor_config.json");

  // Extract necessary parameters from the configuration
  const input_size = pre.size.width; // Assuming width and height are the same for resizing
  const imageMean = pre.image_mean;
  const imageStd = pre.image_std;

  // Create an Image object and load the base64 encoded image
  const img = new Image();
  img.src = base64Data;

  // Create a canvas for image processing
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  // Ensure the canvas size matches the image size
  canvas.width = img.width;
  canvas.height = img.height;

  // Draw the image onto the canvas
  ctx.drawImage(img, 0, 0, img.width, img.height);

  // Get the image data from the canvas
  const imageData = ctx.getImageData(0, 0, img.width, img.height);

  // Resize the image data
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

  // Prepare data in Float32Array format
  const dataFromImage = new Float32Array(3 * input_size * input_size);

  // Normalize and flatten the resized image data
  for (let i = 0; i < input_size * input_size; i++) {
    dataFromImage[i] =
      (resizedImageData.data[i * 4] / 255.0 - imageMean[0]) / imageStd[0]; // R
    dataFromImage[i + input_size * input_size] =
      (resizedImageData.data[i * 4 + 1] / 255.0 - imageMean[1]) / imageStd[1]; // G
    dataFromImage[i + 2 * input_size * input_size] =
      (resizedImageData.data[i * 4 + 2] / 255.0 - imageMean[2]) / imageStd[2]; // B
  }

  // Create an input Tensor for further processing (example with ort.js syntax)
  const inputTensor = new ort.Tensor("float32", dataFromImage, [
    1,
    3,
    input_size,
    input_size,
  ]);

  // Return the processed data or Tensor
  return inputTensor;
}

export async function predict(base64Data) {
  const session = loadInferenceSession();
  const image = await preprocessImage(base64Data);

  return image;
}

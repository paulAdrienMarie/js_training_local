const status = document.getElementById("status");
const fileUpload = document.getElementById("file-upload");
const imageContainer = document.getElementById("image-container");
const input_size = 224;

// read the uploaded file
fileUpload.addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (!file) {
    return;
  }

  const reader = new FileReader();

  // Set up a callback when the file is loaded
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

// make a request to the python server to generate a caption of the given image
function detect(image) {
  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ input_image: image.src }),
  })
    .then((response) => response.json())
    .then((data) => {
      displayOutput(data);
    })
    .catch((error) => {
      document.getElementById("result").textContent = "Error: " + error;
    });
}

function displayOutput(data) {
  console.log(data);
  const labels = data.object_labels;

  const generated_text = document.createElement("div");
  generated_text.id = "textarea-id";

  // Create a container for the labels
  const labelsContainer = document.createElement("div");
  labelsContainer.id = "labels-container";

  // Iterate over the dictionary and create checkboxes
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

  // Append the labels container to the generated text div
  generated_text.appendChild(labelsContainer);

  // Append the generated text to the image container
  const imageContainer = document.getElementById("image-container");
  imageContainer.appendChild(generated_text);

  // Call the displayButtons function (assuming it exists)
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

// Function to add event listener to buttons
function add_Event(id) {
  if (id === "button-retrain-id") {
    document.getElementById(id).addEventListener("click", function () {
      console.log("button-clicked");
      
      const checkedCheckbox = document.querySelector(
        "#labels-container input[type='checkbox']:checked"
      );
      if (checkedCheckbox) {
        // Get the label associated with the checked checkbox
        const labelElement = document.querySelector(
          `label[for="${checkedCheckbox.id}"]`
        );
        const labelText = labelElement.innerText.split(":")[0]; // Extract label text before the col
        // Launch the training request with the label of the checked checkbox
        launch_training_request(labelText);
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

function launch_training_request(new_class) {

  let image = document.getElementById("image-id");

  document.getElementById("result").textContent =
    "Training started with success";

  fetch("/train", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ input_image: image.src, correct_class: new_class }),
  })
    .then((response) => response.json())
    .then((data) => {
      console.log(data);
      document.getElementById("result").textContent = data.message;
    })
    .catch((error) => {
      document.getElementById("result").textContent = "Error: " + error;
    });
}

let scriptTag = document.getElementById("data-script");
let processTextUrl = scriptTag.getAttribute("data-url");

function convertImageUrl(url) {
  let baseUrl = "https://storage.googleapis.com/interactive-static-2/";
  return baseUrl + url;
}

function updateData(data) {
  // Delete all prompt containers first
  let resetContainer = document.getElementById("resetContainer");
  resetContainer.innerHTML = "";
  let promptsContainer = document.getElementById("promptsContainer");
  promptsContainer.innerHTML = "";

  var taskTextElement = document.getElementById("task-text");
  if (data.dataset_index === 0) {
    taskTextElement.innerHTML =
      '<b>Task 1/3: Colored Sprites</b><br>An AI model was trained to classify images of sprites by shape: "square" or "oval". You will be providing feedback on the model errors within "<b>square</b>" images.';
  } else if (data.dataset_index === 1) {
    taskTextElement.innerHTML =
      '<b>Task 2/3: Bird species</b><br> An AI model was trained to classify images of birds by species: "waterbird" or "landbird". You will be providing feedback on the model errors in "<b>waterbird</b>" images.';
  } else if (data.dataset_index === 2) {
    taskTextElement.innerHTML =
      '<b>Task 3/3: Blond vs Non-blond</b><br> An AI model was trained to classify images of people by hair color: "blond" or "non-blond". You will be providing feedback on the model errors within "<b>blond</b>" images.';
  }

  let counter = 0;
  // Update the prompts
  data.all_prompts
    .sort((a, b) => {
      return (
        data.prompt_data[b]["prompt_score"] -
        data.prompt_data[a]["prompt_score"]
      );
    })
    .forEach((prompt) => {
      let listItem = document.createElement("li");
      let score = data.prompt_data[prompt]["prompt_score"];
      listItem.className =
        "list-group-item d-flex justify-content-between align-items-center hover-effect";
      let prompt_text = prompt;
      if (counter === 0) {
        prompt_text += " 🏆";
      }
      counter++;
      listItem.innerHTML = `<span>${prompt_text}</span><span class="badge badge-primary badge-pill">${score}</span>`;
      listItem.addEventListener("click", function () {
        let userText = prompt;
        fetch("/process_text", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            user_text: userText,
            prolificID: output["demographics"]["prolific"],
          }),
        })
          .then((response) => response.json())
          .then((data) => {
            console.log("Data from /process_text:", data);
            updateData(data);
          })
          .catch((error) => {
            console.error("Error:", error);
          });
      });
      promptsContainer.appendChild(listItem);

      let resetButton = document.createElement("button");
      resetButton.className = "btn btn-secondary btn-sm";
      resetButton.innerHTML = "← See original data";
      resetButton.addEventListener("click", function () {
        fetch("/show_initial_data", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        })
          .then((response) => response.json())
          .then((data) => {
            console.log("Data from /show_initial_data", data);
            updateData(data);
          })
          .catch((error) => {
            console.error("Error:", error);
          });
      });
      resetContainer.innerHTML = "";
      resetContainer.appendChild(resetButton);
    });

  let selected_results;
  let text1 = document.getElementById("grid-text-1");
  let text2 = document.getElementById("grid-text-2");
  let grid1 = document.getElementById("image-grid-1");
  let grid2 = document.getElementById("image-grid-2");
  var prompt_list_text_element = document.getElementById("prompt-list-text");

  // Check if data has "prompt"
  if (!("prompt" in data)) {
    console.log("No prompt");
    selected_results = data.default_data["selected_results"];
    prompt_list_text_element.innerHTML = "";
  } else {
    if (
      data.hasOwnProperty("show_initial_data") &&
      data.show_initial_data === true
    ) {
      selected_results = data.default_data["selected_results"];
    } else {
      let c_prompt = data.prompt; // current prompt
      selected_results = data.prompt_data[c_prompt]["selected_results"];
      console.log("Displaying split for prompt: " + c_prompt);
    }

    let good_score_text;
    if (data.dataset_index === 0) {
      good_score_text =
        "A good score for this task is <b>0.6</b>. Try out 'red', and compare with other prompts.";
    } else if (data.dataset_index === 1) {
      good_score_text = "Aim to achieve a score above <b>0.5</b>.";
    } else if (data.dataset_index === 2) {
      good_score_text = "Aim to achieve a score above <b>0.3</b>.";
    }
    var prompt_text =
      "<b>Results:</b><br>You will see the percentage of images the AI is correct on for images with and without the phrase. Click on a phrase to see the corresponding split. We sort the phrases by a score between 0~1 which indicates how well the prompt is splitting the data. " +
      good_score_text +
      "<br>";

    prompt_list_text_element.innerHTML = prompt_text;
  }

  console.log(selected_results);

  text1.innerText = selected_results[0]["text"];
  text2.innerText = selected_results[1]["text"];

  let images1 = selected_results[0]["images"];
  let images2 = selected_results[1]["images"];
  images1 = images1.map((url) => convertImageUrl(url));
  images2 = images2.map((url) => convertImageUrl(url));

  let grid1Images = grid1.getElementsByTagName("img");
  for (let i = 0; i < grid1Images.length; i++) {
    grid1Images[i].src = "";
    grid1Images[i].style.border = "none";
  }
  for (let i = 0; i < Math.min(images1.length, grid1Images.length); i++) {
    grid1Images[i].src = images1[i];
    grid1Images[i].className += " zoom-on-hover";

    if (selected_results[0] && "correct" in selected_results[0]) {
      if (selected_results[0]["correct"][i] === true) {
        // grid1Images[i].style.border = "3px solid rgba(0, 200, 0, 0.7)";
        grid1Images[i].style.border = "none";
      } else if (selected_results[0]["correct"][i] === false) {
        grid1Images[i].style.border = "3px solid rgba(255, 0, 0, 0.6)";
      }
    } else {
      grid1Images[i].style.border = "none";
    }
  }

  let grid2Images = grid2.getElementsByTagName("img");
  for (let i = 0; i < Math.min(images2.length, grid2Images.length); i++) {
    grid2Images[i].src = "";
    grid2Images[i].style.border = "none";
  }
  for (let i = 0; i < Math.min(images2.length, grid2Images.length); i++) {
    grid2Images[i].src = images2[i];
    grid2Images[i].className += " zoom-on-hover";

    if (selected_results[1] && "correct" in selected_results[1]) {
      if (selected_results[1]["correct"][i] === true) {
        // grid2Images[i].style.border = "3px solid rgba(0, 200, 0, 0.7)";
        grid2Images[i].style.border = "none";
      } else if (selected_results[1]["correct"][i] === false) {
        grid2Images[i].style.border = "3px solid rgba(255, 0, 0, 0.6)";
      }
    } else {
      grid2Images[i].style.border = "none";
    }
  }
}

document
  .getElementById("textForm")
  .addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent the form from submitting normally

    let userText = document.getElementById("userText").value;
    console.log(userText);
    if (userText == "") {
      console.log("Prompt cannot be empty!");
      alert("Prompt cannot be empty!");
      return;
    }
    if (
      userText == "square" ||
      userText == "oval" ||
      userText == "waterbird" ||
      userText == "landbird" ||
      userText == "waterbirds" ||
      userText == "landbirds" ||
      userText == "blond" ||
      userText == "blonde" ||
      userText == "non-blond"
    ) {
      console.log("Prompts that describe the task itself are not allowed.");
      alert("Prompts that describe the task itself are not allowed.");
      return;
    }
    fetch("/process_text", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_text: userText,
        prolificID: output["demographics"]["prolific"],
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        updateData(data);
        document.getElementById("userText").value = "";
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  });

window.onload = function () {
  fetch("/init_data", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((response) => response.json())
    .then((data) => {
      console.log(data);
      updateData(data);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
};

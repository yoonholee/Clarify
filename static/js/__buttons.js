// Function to handle active button setting
const setActiveButton = (buttonClass) => {
  $(buttonClass).on("click", function () {
    $(buttonClass).removeClass("active");
    $(this).addClass("active");
  });
};

// Function to handle POST requests and page reload
const sendPostRequest = (endpoint, data, errorMsg) => {
  $.post(endpoint, data, (responseData, status) => {
    if (status === "success") {
      location.reload();
    } else {
      console.error(`${errorMsg} ${status}`);
    }
  });
};

// Dataset selector
$(document).ready(() => setActiveButton(".dataset-button"));

// Reload button
$("#reset-button").on("click", function () {
  sendPostRequest("/reset", {}, "Error: ");
});

// Select class button
$(document).ready(() => {
  $(".class-button").click(function () {
    let class_name = $(this).data("class");
    sendPostRequest("/select_class", { class_name }, "Error selecting class: ");
  });
});

// Toggle buttons for the prompts
$(document).ready(() => {
  setActiveButton(".toggle-button");
  $("body").on("click", ".toggle-button", function () {
    let prompt = $(this).text();
    sendPostRequest("/process_button", { prompt: prompt }, "Error: ");
  });
});

// $('#reset-button').on('click', function () {
//     $('.toggle-button').remove();
// });

$("#add-button").click(function () {
  var selectedClass = $(".class-button.active").data("class");
  var selectedPrompt = $(".toggle-button.active").text();

  $.ajax({
    type: "POST",
    url: "/add_combination",
    data: JSON.stringify({
      class_name: selectedClass,
      prompt: selectedPrompt,
    }),
    contentType: "application/json; charset=utf-8",
    dataType: "json",
    success: function () {
      location.reload();
    },
    error: function () {
      // Handle any errors here.
    },
  });
});

// Reweight button
$(document).ready(function () {
  $("#reweight-button").click(function () {
    $.ajax({
      url: "/reweight",
      method: "POST",
      success: function (response) {
        // Handle the response from the Flask app
        console.log(response);
        location.reload();
      },
      error: function (error) {
        // Handle any errors
        console.error(error);
      },
    });
  });
});

function main() {
  enableTask();
  if (index == 0) {
    $("#onboarding").css("display", "block");
    $("#payment").html(payment);
    $("#time").html(time);
  }
}

function transition(current, next) {
  window.scrollTo(0, 0);
  $("#" + current).css("display", "none");
  $("#" + next).css("display", "block");
  window.scrollTo(0, 0);
  index += 1;
}

function transition_task(current, next) {
  window.scrollTo(0, 0);
  $("#" + current).css("display", "none");
  $("#" + next).css("display", "flex");
  $("#" + next).css("align-content", "flex-start");
  $("#" + next).css("justify-content", "space-between");
  window.scrollTo(0, 0);
  index += 1;
}

function consentCallback() {
  startTime = Date.now();
  if ($("#agree").is(":checked")) {
    output["consent"] = "agree";
    transition("onboarding", "demographics");
  } else if ($("#disagree").is(":checked")) {
    output["consent"] = "disagree";
    transition("onboarding", "exit");
  } else {
    alert("Please answer the consent question.");
  }
}

function demographicsCallback() {
  let checked_prolificID = false;
  let written_prolificID;
  if ($.trim($("#prolificID").val())) {
    written_prolificID = $("#prolificID").val();
    checked_prolificID = true;
  }

  let gender = ["#male", "#female", "#nonbinary", "#prefernot", "#other"];
  let checked_gender = false;
  let chosen_gender;
  for (let i = 0; i < gender.length; i++) {
    if ($(gender[i]).is(":checked")) {
      if (gender[i] != "#other") {
        chosen_gender = gender[i].slice(1);
        checked_gender = true;
        break;
      } else if ($.trim($("#self-description").val())) {
        chosen_gender = $("#self-description").val();
        checked_gender = true;
        break;
      }
    }
  }
  let checked_age = false;
  let written_age;
  if ($.trim($("#age").val())) {
    written_age = $("#age").val();
    checked_age = true;
  }
  if (checked_prolificID == false) {
    alert("Please answer the question about your ID.");
  }
  if (checked_gender == false) {
    alert("Please answer the question about your gender.");
  }
  if (checked_age == false) {
    alert("Please answer the question about your age.");
  }
  if (
    checked_gender == true &&
    checked_age == true &&
    checked_prolificID == true
  ) {
    output["demographics"] = {
      prolific: written_prolificID,
      gender: chosen_gender,
      age: written_age,
    };
    transition("demographics", "introduction");
  }
}
function introductionCallback() {
  transition_task("introduction", "task");
}
function taskCallback() {
  console.log("pressed continue");
  var numPrompts = $("#promptsContainer").children().length;
  if (numPrompts < 3) {
    alert("Please try at least 3 prompts before moving on.");
    return;
  }

  $.ajax({
    url: "/continue",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      prolificID: output["demographics"]["prolific"],
    }),
    success: function (response) {
      if (!response.out_of_tasks) {
        console.log("loading next task");
        // Wait for a short delay before making the fetch call
        setTimeout(function () {
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
              // var progress_increment = 1 / 3;
              // progress += progress_increment;
              // $('#progress-bar').attr('aria-valuenow', Math.round(progress)).css('width', Math.round(progress)+'%')
            })
            .catch((error) => {
              console.error("Error:", error);
            });
        }, 500); // Delay of 500 milliseconds
      } else {
        console.log("done with all tasks");
        transition("task", "feedback");
      }
    },
    error: function (error) {
      console.log(error);
    },
  });
}
function feedbackCallback() {
  if (!$.trim($("#feedback1").val())) {
    alert("Please input your response in all text boxes.");
  } else if (!$.trim($("#feedback2").val())) {
    alert("Please input your response in all text boxes.");
  } else if (!$.trim($("#feedback3").val())) {
    alert("Please input your response in all text boxes.");
  } else {
    output["feedback1"] = $("#feedback1").val();
    output["feedback2"] = $("#feedback2").val();
    output["feedback3"] = $("#feedback3").val();
    transition("feedback", "submission");
  }
}
function submissionCallback() {
  console.log("submitting data");
  console.log(output);
  $.ajax({
    url: "/logData",
    type: "POST",
    contentType: "application/json", // Add this line
    data: JSON.stringify({
      // And stringify your data
      prolificID: output["demographics"]["prolific"],
      data: output,
    }),
    success: function () {
      console.log("Data logged successfully");
      window.location.href =
        "https://app.prolific.co/submissions/complete?cc=CVY7K7BT";
    },
    error: function (error) {
      console.error("Error logging data:", error);
    },
  });
}

function enableTask() {
  $("#consent-button").click(consentCallback);
  $("#demographic-button").click(demographicsCallback);
  $("#introduction-button").click(introductionCallback);
  $("#task-button").click(taskCallback);
  $("#feedback-button").click(feedbackCallback);
  $("#submit-btn").click(submissionCallback);
}

function disableSelfDescribe() {
  $("#self-description").attr("disabled", true);
}

main();

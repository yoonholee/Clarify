// Function to compute density
const kernelDensityEstimator = (kernel, X) => (V) =>
  X.map((x) => [x, d3.mean(V, (v) => kernel(x - v)) * V.length]);

// Kernel function
const kernelEpanechnikov = (k) => (v) =>
  Math.abs((v /= k)) <= 1 ? (0.75 * (1 - v * v)) / k : 0;

// Color variables for plot
const colors = {
  correct: "rgba(0, 200, 0, 0.3)",
  wrong: "rgba(255, 0, 0, 0.45)",
};

// Fetch and process data
fetch(`${filename}`)
  .then((response) => {
    if (!response.ok) throw new Error("HTTP error " + response.status);
    return response.json();
  })
  .then((data) => {
    // Parse and calculate data
    const plotData = data;
    const sim_min = d3.min(plotData, (d) => d.Similarity);
    const sim_max = d3.max(plotData, (d) => d.Similarity);
    const sim_range = sim_max - sim_min;
    console.log("Plot data:", plotData);
    console.log("sim_min:", sim_min, "sim_max:", sim_max);

    const full_width = d3
      .select("#main-grid")
      .node()
      .getBoundingClientRect().width;
    const full_scatter_height = 350;
    const hor_margin = 0.1 * full_width;
    const margin = { top: 10, right: hor_margin, bottom: 0, left: hor_margin };
    const width = full_width - margin.left - margin.right;
    const scatter_height = full_scatter_height - margin.top - margin.bottom;
    console.log(`Full width: ${full_width}`);
    console.log(`Width: ${width}, Height: ${scatter_height}`);

    const svgScatter = d3
      .select("#similarities-plot")
      .append("svg")
      .attr("width", full_width)
      .attr("height", full_scatter_height)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Define your scales
    const xScale = d3
      .scaleLinear()
      .range([0, width])
      .domain(d3.extent(data, (d) => d.Similarity));
    console.log("xScale:", xScale);

    const yScale = d3
      .scaleLinear()
      .range([scatter_height, 0])
      .domain([-0.05, 1.05]);
    console.log(d3.extent(data, (d) => d.confidence));
    console.log("yScale:", yScale);

    // Bind data to the SVG elements for the scatter plot
    points = svgScatter
      .selectAll("circle")
      .data(data)
      .enter()
      .append("circle")
      .attr("class", (d) => d.text)
      // .attr("class", d => d.text === 'correct' ? 'correct' : 'wrong')
      .attr("cx", (d) => xScale(d.Similarity))
      .attr("cy", (d) => yScale(d.confidence))
      .attr("r", 4) // radius of the circle
      // .style("fill", d => d.text === 'correct' ? colors.correct : colors.wrong)
      .style("fill", (d) => colors[d.text])
      .style("opacity", 0.6); // semi-transparent

    var yAxis = d3.axisLeft(yScale);
    svgScatter.append("g").call(yAxis);

    // Add y-axis label
    svgScatter
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -margin.left * 0.8) // Adjust this to move the text left/right when looking at the plot upright
      .attr("x", -scatter_height / 2) // position the label in the center of y-axis
      .attr("dy", "1em") // slight adjustment to place text at middle of the axis line.
      .style("text-anchor", "middle") // text will be centered at the position specified by x and y
      .text("Model Confidence");

    // Add interactivity
    const div = d3
      .select("body")
      .append("div")
      .attr("class", "tooltip")
      .style("opacity", 0)
      .style("position", "absolute") // Position absolute
      .style("pointer-events", "none"); // Disable pointer events

    const cdn_root = "https://storage.googleapis.com/interactive-static";
    points
      .on("mouseover", function () {
        div
          .transition()
          .duration(0)
          .style("opacity", 1.0)
          .style("pointer-events", "auto"); // Enable pointer events

        const svgCoords = this.getBoundingClientRect();

        if (this.__data__.filename) {
          console.log(this.__data__.filename);
          div
            .html(
              `<img src="${cdn_root}/${this.__data__.filename}" width='100' height='100'>`
            )
            .style("left", svgCoords.x + "px") // Use SVG coordinates
            .style("top", svgCoords.y + "px");
        } else {
          console.log(this.__data__);
        }
      })
      .on("mouseout", function () {
        div
          .transition()
          .duration(100)
          .style("opacity", 0.5) // Set opacity to 0
          .style("pointer-events", "none") // Disable pointer events
          .on("end", function () {
            // At the end of the transition
            div.html(""); // Clear the tooltip content
          });
      });

    const full_distplot_height = 200;
    const d_margin = {
      top: 10,
      right: hor_margin,
      bottom: 50,
      left: hor_margin,
    };
    const distplot_height =
      full_distplot_height - d_margin.top - d_margin.bottom;

    // Define distplot svg
    const svgDensity = d3
      .select("#similarities-plot")
      .append("svg")
      .attr("width", full_width)
      .attr("height", full_distplot_height)
      .append("g")
      .attr("transform", `translate(${d_margin.left},${d_margin.top})`);

    // Define scales
    const x = d3
      .scaleLinear()
      .range([0, width])
      .domain([sim_min - 0.1 * sim_range, sim_max + 0.1 * sim_range]);
    const yDensity = d3.scaleLinear().range([distplot_height, 0]);
    console.log("Output of x scale given input 0.1: ", x(0.1));
    console.log("Output of yDensity scale given input 0.1: ", yDensity(0.1));

    function processDensity(data) {
      const k = sim_range / 10;
      const kde = kernelDensityEstimator(kernelEpanechnikov(k), x.ticks(50));
      let density = kde(data.map((d) => d.Similarity));

      const sum = density.reduce((acc, cur) => acc + cur[1], 0);
      console.log("Sum:", sum);

      return density.map(([x, y]) => [x, y / sum]);
    }

    const [data1, data2] = ["correct", "wrong"].map((text) =>
      plotData.filter((d) => d.text === text)
    );
    let [density1, density2] = [data1, data2].map((data) =>
      processDensity(data)
    );

    console.log("data1 length:", data1.length, "data2 length:", data2.length);
    // Set y domain and add x and y axes
    yDensity.domain([
      0,
      Math.max(
        d3.max(density1, (d) => d[1]),
        d3.max(density2, (d) => d[1])
      ),
    ]);

    svgDensity
      .append("g")
      .attr("transform", `translate(0,${distplot_height})`)
      .call(d3.axisBottom(x));

    svgDensity
      .append("text") // x-axis label
      .attr("x", width * 0.5)
      .attr("y", distplot_height + d_margin.bottom / 2) // position the label in the center of y-axis
      .attr("dy", "1em") // slight adjustment to place text at middle of the axis line.
      .style("text-anchor", "middle") // text will be centered at the position specified by x and y
      .text("Similarity");

    svgDensity
      .append("text") // y-axis label
      .attr("transform", "rotate(-90)")
      .attr("y", -d_margin.left * 0.8) // Adjust this to move the text left/right when looking at the plot upright
      .attr("x", -distplot_height / 2) // position the label in the center of y-axis
      .attr("dy", "1em") // slight adjustment to place text at middle of the axis line.
      .style("text-anchor", "middle") // text will be centered at the position specified by x and y
      .text("Density");

    // Function for plotting densities
    const plotDensity = (data, color) =>
      svgDensity
        .append("path")
        .datum(data)
        .attr("fill", color)
        .attr("opacity", ".7")
        .attr("stroke", "#000")
        .attr("stroke-width", 1.5)
        .attr("stroke-linejoin", "round")
        .attr(
          "d",
          d3
            .line()
            .curve(d3.curveBasis)
            .x((d) => x(d[0]))
            .y((d) => yDensity(d[1]))
        );

    // Plot densities
    plotDensity(density1, colors.correct);
    plotDensity(density2, colors.wrong);

    // Add a legend
    const legend = svgDensity
      .selectAll(".legend")
      .data(["Correct", "Wrong"])
      .enter()
      .append("g")
      .attr("class", "legend")
      .attr("transform", (d, i) => `translate(0,${i * 20})`);

    legend
      .append("rect")
      .attr("x", width - 18)
      .attr("width", 18)
      .attr("height", 18)
      .style("fill", (d) => (d === "Correct" ? colors.correct : colors.wrong));

    legend
      .append("text")
      .attr("x", width - 24)
      .attr("y", 9)
      .attr("dy", ".35em")
      .style("text-anchor", "end")
      .text((d) => d);

    // Update the session data
    const updateSession = (event) => {
      const xValue = Math.max(x(sim_min), Math.min(x(sim_max), event.x));
      const domainValue = x.invert(xValue);
      localStorage.setItem("sliderXValue", xValue);

      fetch("/update_slider", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slider_value: domainValue }),
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          location.reload();
        })
        .catch((e) => {
          console.log(
            "There was a problem with the fetch operation: " + e.message
          );
        });
    };

    // Function for when slider is dragged
    const dragged = (event) => {
      const xValue = Math.max(x(sim_min), Math.min(x(sim_max), event.x));
      sliderLine.attr("x1", xValue).attr("x2", xValue);
      localStorage.setItem("sliderXValue", xValue);
    };

    // Add slider
    let xValueStored = localStorage.getItem("sliderXValue");
    const slider_val = xValueStored
      ? parseFloat(xValueStored)
      : x(0.5 * (sim_max + sim_min));

    const sliderLine = svgDensity
      .append("line")
      .attr("x1", slider_val)
      .attr("x2", slider_val)
      .attr("y1", 0)
      .attr("y2", distplot_height)
      .attr("stroke", "black")
      .attr("stroke-width", 2)
      .attr("cursor", "move")
      .call(d3.drag().on("drag", dragged).on("end", updateSession));
  })
  .catch(function () {
    console.log("An error occurred while fetching the JSON data.");
  });

document.addEventListener("DOMContentLoaded", function () {
  const images = document.querySelectorAll(".sim-img");
  images.forEach((img) => {
    const correct = img.getAttribute("data-correct");
    if (correct === "True") {
      img.style.border = "3px solid rgba(0, 200, 0, 0.7)";
    } else if (correct === "False") {
      img.style.border = "3px solid rgba(255, 0, 0, 0.6)";
    } else {
      img.style.border = "none";
    }
  });
});

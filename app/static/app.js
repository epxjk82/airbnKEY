let get_input_latlng = function() {
      let lat = $("input#lat").val()
      let lng = $("input#lng").val()
      let prop_type = $("select#prop_type").val()
      let bath = $("input#bath").is(":checked")
      let view = $("input#view").is(":checked")
      let instabook = $("input#instabook").is(":checked")
      // let reviews = $("input#reviews").val()
      // let rating = $("input#rating").val()
      // let bath = $("select#bath").val()
      // let view = $("select#view").val()
      // let instabook = $("select#instabook").val()
      // let prop_type = $("select#prop_type").val()

      return {'lat': parseFloat(lat),
              'lng': parseFloat(lng),
              'prop_type': prop_type,
              'bath': bath,
              'view': view,
              'instabook': instabook,
              'prop_type': prop_type,
              // 'reviews': parseFloat(reviews),
              // 'rating': parseFloat(rating),
             }
};

// Set the dimensions of the canvas / graph
var margin = {top: 20, right: 20, bottom: 70, left: 40},
    width = 450 - margin.left - margin.right,
    height = 300 - margin.top - margin.bottom,
    barHeight= 20;

// Set the ranges
var x = d3.scale.linear()
          .range([0,width]);

x.domain([0,120])

var chart = d3.select(".chart")
              .attr("width",width)
              .attr("height", barHeight*12);

// Getting default data stored in csv
function showDefault() {
    // document.getElementById("rating_val").innerHTML = document.getElementById("rating").value;
    d3.csv("static/pred.csv", type, updateChart);
}

function updateNew(data) {
    d3.json(data, updateChart);
}

showDefault();

function type(d) {
    d.prediction = +d.prediction;
    return d;
}

function updateChart(data) {
    // y.domain([0, d3.max(data, function(d) { return d.value; })]);
    // y-axis
    // chart.select(".y.axis").remove();
    // chart.append("g")
    //       .attr("class", "y axis")
    //       .call(yAxis)
    //   .append("text")
    //     .attr("transform", "rotate(-90)")
    //     .attr("y", 6)
    //     .attr("dy", ".71em")
    //     .style("text-anchor", "end")
    //     .text("Frequency");

    var bar = chart.selectAll("g")
                   .data(data)
                   .enter()
                   .append("g")
                   .attr("transform",function(d,i){
                     return "translate(0,"+i*barHeight+")"
                   });

    rect = bar.append("rect")
      .attr("width","1")
      .attr("height",barHeight-1).attr("padding","5")
      .attr("class", function(d){return d.month} );

    bar.append("text")
      //.attr("x",function(d){return x(d.prediction)+5})
      .attr("x",5)
      .attr("y",barHeight/2)
      .attr("dy",".35em").text(function(d){return d.month});

    bar.append("text")
      .attr("x",function(d){return x(d.prediction)-5})
      .attr("y",barHeight/2)
      .attr("dy",".35em").text(function(d){return d.prediction})
      .style("text-anchor","end");

    rect.transition()
      .attr("width",function(d){return x(d.prediction);})
      .duration(750);


};

let make_prediction = function(latlng) {
      $.ajax({
          url: '/predict',
          contentType: "application/json; charset=utf-8",
          type: 'POST',
          data: JSON.stringify(latlng),
          success: function (data) {
              //display_prediction(data);
              //$("img#monthly_inc").toggle();
              //$("span#predict").html(data)

              // updateNew(data);
              // d3.selectAll('rect').transition()
              //   .attr("width",function(d){return x(d.prediction);})
              //   .duration(1000)

              $(".chart").empty()
              x.domain([0,120])

              var bar = chart.selectAll("g")
                             .data(data)
                             .enter().append("g")
                             .attr("transform",function(d,i){return "translate(0,"+i*barHeight+")"});

              rect = bar.append("rect")
                        .attr("width","1").attr("height",barHeight-1).attr("padding","5");

              rect.transition()
                  .attr("width",function(d){return x(d.prediction);}).duration(0);

              bar.append("text")
                 //.attr("x",function(d){return x(d.prediction)+5})
                 .attr("x",5)
                 .attr("y",barHeight/2)
                 .attr("dy",".35em").text(function(d){return d.month});

              bar.append("text")
                 .attr("x",function(d){return x(d.prediction)-5})
                 .attr("y",barHeight/2)
                 .attr("dy",".35em").text(function(d){return d.prediction})
                 .style("text-anchor","end");


          }
      });
};

$(document).ready(function() {

      $("button#predict").click(function() {

          make_prediction(get_input_latlng());

      });

      $("select.dropdown").change(function() {

          make_prediction(get_input_latlng());

      });

      $("input.switch").click(function() {

          make_prediction(get_input_latlng());

      });

      // $("input#reviews").click(function() {
      //     make_prediction(get_input_latlng());
      //     document.getElementById("reviews_val").innerHTML = document.getElementById("reviews").value;
      // });
      //
      // $("input#rating").click(function() {
      //     make_prediction(get_input_latlng());
      //     document.getElementById("rating_val").innerHTML = document.getElementById("rating").value;
      // });
});

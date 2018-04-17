d3.json("http://localhost:8000/es_tool/json/brands_sim.json").then(function(result){
	data = result['brands_sim'];
	//var svg = d3.select("#scatter").append("svg");
	//width = parseInt(window.getComputedStyle(document.getElementById("scatter")).getPropertyValue("width").slice(0,-2));
	//height =  parseInt(window.getComputedStyle(document.getElementById("scatter")).getPropertyValue("height").slice(0,-2));
	//svg.attr("width", width);
	//svg.attr("height", height);

//	svg.attr("padding-right", "19px")
var margin = {
	 top: 0,
	 right: -15,
	 bottom: 0,
	 left: -15
 },
  width = document.querySelector('#scatter').offsetWidth - margin.left - margin.right,
  height = document.querySelector('#scatter').offsetHeight - margin.top - margin.bottom;


// add the graph canvas to the body of the webpage
var svg = d3.select("#scatter").append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
	.attr("viewBox", "0 -67 " +(parseInt(height)+40)+" "+(parseInt(width)+100))
  //.attr("id", "scatter")

  //.attr("class", "graphSvgComponent")
  .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
	var x = d3.scaleBand().rangeRound([0, width]).padding(0.1),
		y = d3.scaleLinear().rangeRound([height, 0]);

	var g = svg.append("g");

	x.domain(data.map(function(d) { return d.name; }));
	y.domain([0, d3.max(data, function(d) { return d.sim; })]);



	//g.append("g")
	  //.attr("class", "axis axis--y")
	  //.call(d3.axisLeft(y).ticks(5, "%"))
	//.append("text")
	  //.attr("transform", "rotate(-90)")
	  //.attr("y", 6)
	  //.attr("dy", "0.71em")
	  //.attr("text-anchor", "end")
	  //.text("Brand");


	g.selectAll(".bar")
	.data(data)
	.enter().append("circle")
	  .attr("class", "bar")
		.attr("id", function(d,i){return "dot"+i})
	  .attr("cx", function(d) { return x(d.name); })
	  .attr("cy", function(d) { return y(d.sim); })
	  .attr("width", x.bandwidth())
	  //.attr("height", function(d) { return height - y(d.frequency); });
	  .attr("r", "0.7em")
		.attr("fill", function(d){return "rgba(238, 54, 15, "+d.sim/100+")";})
		.attr("stroke", "#578290");
		//.style("opacity", function(d) {return d.sim/100;});


		g.selectAll("text")
		  .data(data)
			.enter().append("text")
			.attr("class", "labels")
			.attr("class", "opacity")
			.attr("id", function(d,i){return "label"+i})
			.attr("x", function(d) { return x(d.name)-10; })
			.attr("y", function(d) { return y(d.sim)-25; })
			.text(function(d) {
				return d.name;
			})
			.attr("font-family", "Karla")
			.attr("font-size","0.8em")
			.attr("fill", "#578290");


			g.append("g")
			.attr("class", "axis axis--x")
			.attr("transform", "translate(0," + height + ")")
			.call(d3.axisBottom(x));

			g.append("text")
			.attr("class", "scatter-title")
			.attr("x",+(parseInt(width)/15))
			.attr("y", +(parseInt(height)+30))
			.text("BRANDS POSITIONED ACCORDING TO THEIR RELEVANCE")

});


//Here jQuery animation for scatterplot
$(function() {
  $('#dot0').hover(function() {
    $('#label0').removeClass('opacity');
  }, function() {
    $('#label0').addClass('opacity');
  });
});

$(function() {
  $('#dot1').hover(function() {
    $('#label1').removeClass('opacity');
  }, function() {
    $('#label1').addClass('opacity');
  });
});

$(function() {
  $('#dot2').hover(function() {
    $('#label2').removeClass('opacity');
  }, function() {
    $('#label2').addClass('opacity');
  });
});

$(function() {
  $('#dot3').hover(function() {
    $('#label3').removeClass('opacity');
  }, function() {
    $('#label3').addClass('opacity');
  });
});

$(function() {
  $('#dot4').hover(function() {
    $('#label4').removeClass('opacity');
  }, function() {
    $('#label4').addClass('opacity');
  });
});

$(function() {
  $('#dot5').hover(function() {
    $('#label5').removeClass('opacity');
  }, function() {
    $('#label5').addClass('opacity');
  });
});

$(function() {
  $('#dot6').hover(function() {
    $('#label6').removeClass('opacity');
  }, function() {
    $('#label6').addClass('opacity');
  });
});

$(function() {
  $('#dot7').hover(function() {
    $('#label7').removeClass('opacity');
  }, function() {
    $('#label7').addClass('opacity');
  });
});

$(function() {
  $('#dot8').hover(function() {
    $('#label8').removeClass('opacity');
  }, function() {
    $('#label8').addClass('opacity');
  });
});



d3.json("http://localhost:8000/es_tool/json/caption.json").then(function(dataset){
	width = parseInt(window.getComputedStyle(document.getElementById("caption")).getPropertyValue("width").slice(0,-2));
	height =  parseInt(window.getComputedStyle(document.getElementById("caption")).getPropertyValue("height").slice(0,-2));

	var colors = []
    for(i=0; i<dataset.children.length; i++){
      colors.push("#"+((1<<24)*Math.random()|0).toString(16));
    };
	var bubble = d3.pack(dataset)
		.size([height, width])
		.padding(1.05);

	var svg = d3.select("#caption")
		.append("svg");
	svg.attr("viewBox", "100 100 "+width+" "+height);
  svg.attr("id","bubblechart")
	var stoca = svg.append("g").attr("class", "stocazzo");
	var nodes = d3.hierarchy(dataset)
		.sum(function(d) { return d.score*2; });

	var node = stoca.selectAll(".node")
		.data(bubble(nodes).descendants())
		.enter()
		.filter(function(d){
			return  !d.children
		})
		.append("g")
		.attr("class", "node")
		.attr("transform", function(d) {
			return "translate(" + d.x*2 + "," + d.y*2 + ")";
		});

	node.append("title")
		.text(function(d) {
			return d.name + ": " + d.score;
		});

	node.append("circle")
		.attr("r", function(d) {
			return d.r*2;
		})
		.attr("fill", function(d){return "rgba(255, 191, 18, "+d.data.score/100+")";})

	node.append("text")
		.attr("dy", ".1em")
		.style("text-anchor", "middle")
		.text(function(d) {
			return d.data.topic;
		})
		.attr("font-family", "Karla")
		.attr("font-size", function(d){
			return d.r/2;
		})
		.attr("fill", "white");

	node.append("text")
		.attr("dy", "1.3em")
		.style("text-anchor", "middle")
		.text(function(d) {
			return d.data.score;
		})
		.attr("font-family",  "Karla")
		.attr("font-size", function(d){
			return d.r/3;
		})
		.attr("fill", "white");

});


//tags
d3.json("http://localhost:8000/es_tool/json/tags.json").then(function(result){
	data = result['tags'];
	var tags = d3.select("#tags");
	tags.selectAll(".tag")
	.data(data)
	.enter().append("div")
	  .attr("class", "tag")
	  .text(function(d){return d.tag;})
		.attr("fill", "white")
      .style("background-color", function(d){return "rgba(255, 140, 18, "+d.score/100+")";});
});

var h = window.getComputedStyle(document.getElementById("squid")).getPropertyValue("height").slice(0, -2);
var w = window.getComputedStyle(document.getElementById("squid")).getPropertyValue("width").slice(0, -2);

$("#squid").attr("viewBox", -w / 2 + " " + -h / 2 + " " + w + " " + h);

prepare_squid = function (target) {
	url = "http://localhost:8000/es_tool/json/clusters.json"
	d3.json(url).then(function (clusters_file) {
		var result = clusters_file.clusters.filter(function (squid) {
			return squid.center.name === target.replace("_", " ")
		})[0]
		//some variables needed for the viz
		var nodes = result.satellites;
		var colors = []
		for (i = 0; i < nodes.length; i++) {
			colors.push("#" + ((1 << 24) * Math.random() | 0).toString(16));
		};
		var center_color = "#" + ((1 << 24) * Math.random() | 0).toString(16);
		var splits = (2 * Math.PI) / nodes.length;
		var squid = d3.select("#squid");
		var all = squid.append("g").attr("class", "all");

		//empty the svg
		//squid.selectAll("*").remove();

		//add links to svg (only if empty)
		console.log(result)

		var scaled_distance = d3.scaleLinear()
		.domain([0, d3.max(result.satellites, function (d) {
			return d.distance;
		})])
		.range([0, w/2])
		
		var links = all.selectAll("line")
			.data(nodes)
			.enter()
			.append("line")
			.attr("i", function (d, i) {
				return i;
			})
			.attr("class", "links")
			.attr("stroke-width", 2)
			.attr("x1", function (d, i) {
				var theta = i * splits + Math.PI / 2;
				return (result.center.size / 20) * Math.cos(theta);
			})
			.attr("y1", function (d, i) {
				var theta = i * splits + Math.PI / 2;
				return (result.center.size / 20) * Math.sin(theta);
			})
			.style("stroke", "#99aeb5");

		//add satellites circles to svg (only if empty)
		var circles = all
			.selectAll("circle")
			.data(nodes)
			.enter()
			.append("circle")
			.attr("i", function (d, i) {
				return i;
			})
			.attr("id", function (d) {
				return d.name.replace(" ", "_");
			})
			.attr("onclick", function (d) {
				return "target_change('" + d.name.replace(" ", "_") + "')";
			})
			.attr("class", "satellite")
			.attr("r", function (d) {
				return d.size / 20;
			})
			.attr("fill", function (d) {
				return d.fill;
			})
			.style("stroke","#4d7a89");

		//add center circle to svg
		d3.select(".all")
			.append("circle")
			.attr("class", "center")
			.attr("id", result.center.name.replace(" ", "_"))
			.attr("cy", 0)
			.attr("cx", 0)
			.attr("r", result.center.size / 20)
			.attr("fill", result.center.fill)
			.style("stroke","#4d7a89");

		//statellites transitions
		circles.data(nodes)
			.transition()
			.duration(2000)
			.attr("cy", function (d, i) {
				var theta = i * splits + Math.PI / 2;
				var rc = result.center.size / 20;
				var d = scaled_distance(d.distance);
				var rs = parseInt(d3.select(this).attr("r"));
				return (rc + d + rs) * Math.sin(theta);
			})
			.attr("cx", function (d, i) {
				var theta = i * splits + Math.PI / 2;
				var rc = result.center.size / 20;
				var d = scaled_distance(d.distance);
				var rs = parseInt(d3.select(this).attr("r"));
				return (rc + d + rs) * Math.cos(theta);
			});

		//links transitions
		links.data(nodes)
			.transition()
			.duration(2000)
			.attr("y2", function (d, i) {
				var theta = i * splits + Math.PI / 2;
				var rc = result.center.size / 20;
				var d = scaled_distance(d.distance);
				return (rc + d) * Math.sin(theta);
			})
			.attr("x2", function (d, i) {
				var theta = i * splits + Math.PI / 2;
				var rc = result.center.size / 20;
				var d = scaled_distance(d.distance);
				return (rc + d) * Math.cos(theta);
			});

		//SVG filter for the gooey effect
		//Based on http://tympanus.net/codrops/2015/03/10/creative-gooey-effects/
		var filter = squid.append("defs")
			.append("filter")
			//use a unique id to reference again later on
			.attr("id", "gooeyCodeFilter");

		//Append multiple "pieces" to the filter
		filter.append("feGaussianBlur")
			.attr("in", "SourceGraphic")
			.attr("stdDeviation", "25")
			.attr("color-interpolation-filters", "sRGB")
			.attr("result", "blur");
		filter.append("feColorMatrix")
			//the class used later to transition the gooey effect
			.attr("class", "blurValues")
			.attr("in", "blur")
			.attr("mode", "matrix")
			.attr("values", "1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 25 -9")
			.attr("result", "gooey");

		//If you want the end shapes to be exactly the same size as without
		//the filter add the feBlend below. However this will result in a
		//less beautiful gooey effect
		filter.append("feBlend")
			.attr("in", "SourceGraphic")
			.attr("in2", "gooey");

		//Apply the filter to the group element of all the circles
		//		var circleWrapper = d3.select(".all")
		//			.style("filter", "url(#gooeyCodeFilter)");
	});
};

function plot_squid(target) {
	url = "http://localhost:8000/es_tool/json/clusters.json"
	d3.json(url).then(function (clusters_file) {
		var result = clusters_file.clusters.filter(function (squid) {
			return squid.center.name === target.replace("_", " ")
		})[0]
		var scaled_distance = d3.scaleLinear()
		.domain([0, d3.max(result.satellites, function (d) {
			return d.distance;
		})])
		.range([0, w/2])
				
		var nodes = result.satellites;
		var splits = (2 * Math.PI) / nodes.length;
		var squid = d3.select("#squid");

		var old_center_DOM = d3.select(".center");
		var old_center = nodes.filter(function (d) {
			return d.name == old_center_DOM.attr("id").replace("_", " ");
		})[0];
		var new_center = result.center;
		var new_center_DOM = d3.select("#" + new_center.name.replace(" ", "_"));

		var new_theta = (new_center_DOM.attr("i")) * splits + Math.PI / 2;
		var rc = result.center.size / 20;
		var d = old_center.distance * (w / 2);
		var rs = parseInt(old_center_DOM.attr("r"));
		var new_cy = (rc + d + rs) * Math.sin(new_theta);
		var new_cx = (rc + d + rs) * Math.cos(new_theta);

		//switch_target
		old_center_DOM.classed("center", false);
		old_center_DOM.classed("satellite", true);
		old_center_DOM.attr("i", new_center_DOM.attr("i"));
		old_center_DOM.attr("onclick", function (d) {
			return "target_change('" + old_center.name.replace(" ", "_") + "')"
		});
		old_center_DOM.transition()
			.duration(2000)
			.attr("cy", new_cy)
			.attr("cx", new_cx);
		//.attr("fill", "blue");

		new_center_DOM.classed("center", true);
		new_center_DOM.classed("satellite", false);
		new_center_DOM.attr("i", "");
		new_center_DOM.attr("onclick", "");
		new_center_DOM.transition()
			.duration(2000)
			.attr("cy", 0)
			.attr("cx", 0);
		//.attr("fill", "red");

		//some variables needed for the viz
		var circles = d3.selectAll(".satellite");
		var links = d3.selectAll(".links");

		//statellites transitions
		circles.data(nodes, function (d) {
				return d ? d.name : this.id.replace("_", " ");
			})
			.transition()
			.duration(2000)
			.attr("cy", function (d) {
				var theta = d3.select(this).attr("i") * splits + Math.PI / 2;
				var rc = result.center.size / 20;
				var d = scaled_distance(d.distance);
				var rs = parseInt(d3.select(this).attr("r"));
				return (rc + d + rs) * Math.sin(theta);
			})
			.attr("cx", function (d) {
				var theta = d3.select(this).attr("i") * splits + Math.PI / 2;
				var rc = result.center.size / 20;
				var d = scaled_distance(d.distance);
				var rs = parseInt(d3.select(this).attr("r"));
				return (rc + d + rs) * Math.cos(theta);
			});

		//links transitions
		setTimeout(function () {
		links.attr("y1", function () {
				var i = d3.select(this).attr("i");
				var theta = i * splits + Math.PI / 2;
				return rc * Math.sin(theta);
			})
			.attr("x1", function () {
				var i = d3.select(this).attr("i");
				var theta = i * splits + Math.PI / 2;
				return rc * Math.cos(theta);
			})
			.attr("y2", function () {
				var i = d3.select(this).attr("i");
				var theta = i * splits + Math.PI / 2;
				var d = scaled_distance(circles.filter("[i='" + i + "']").data()[0].distance);
				return (d + rc) * Math.sin(theta);
			})
			.attr("x2", function () {
				var i = d3.select(this).attr("i");
				var theta = i * splits + Math.PI / 2;
				var d =  scaled_distance(circles.filter("[i='" + i + "']").data()[0].distance);
				return (d + rc) * Math.cos(theta);
			});}, 100);

			links.transition()
			.duration(100)
			.style("opacity", 0);
			links.transition()
			.duration(500)
			.delay(1500)
			.style("opacity", 1);
			d3.select("#dropdownMenuButton").style("border", "2px solid " + result.center.fill)
	});
}

plot_pictures = function () {
	url = "http://localhost:8000/get_scored_pics"
	d3.json(url).then(function (result) {
		data = result.scores
		d3.selectAll(".pics")
			.transition()
			.duration(200)
			.attr("r", 0)
		setTimeout(function()  { d3.selectAll(".pics").remove() }, 200);
		
		setTimeout(function () {

			d3.selectAll(".pics").remove();
		}, 200);
		var communities = d3.selectAll("circle");
		var pic_tooltip;
		if(d3.select("#pic_tooltip").empty()){
			pic_tooltip = d3.select(".all").append("text").attr("id", "pic_tooltip");
		} else {
			pic_tooltip = d3.select("#pic_tooltip");
		}
		var simulations = []
		var tick_closures = {}
		setTimeout(function () {

			var coll_sim = d3.forceSimulation(data).force("charge", d3.forceCollide().radius(10))
			communities.each(
				function () {
					var comm = {
						r: parseInt(d3.select(this).attr("r")),
						cx: parseInt(d3.select(this).attr("cx")),
						cy: parseInt(d3.select(this).attr("cy")),
						name: d3.select(this).attr("id"),
						fill: d3.select(this).attr("fill")
					}
					comm_pics = data.filter(function (d) {
						return d.community === comm.name
					})
					nodes = d3.select(".all")
						.selectAll("." + comm.name + "_pics")
						.data(comm_pics)
						.enter()
						.append("circle")
						.attr("class", comm.name + "_pics pics")
						.attr("r", "5")
						.attr("fill", comm.fill)
						.attr("stroke", "#4d7a89")
						.on("mouseover", function (d) {
							pic_tooltip
								.attr("x", parseInt(d3.select(this).attr("cx")) + 8)
								.attr("y", parseInt(d3.select(this).attr("cy")) - 8)
								.style("fill", "#4d7a89")
								.style("background-color", "red")
								.style("font-size", "11")
								.style("display", "block")
								.text(d.pic_url)
							d3.select("#post_" + d.pic_url.replace(".", "_")).style("border", "2px solid " + comm.fill)
						})
						.on("mouseout", function (d) {
							pic_tooltip.style("display", "none")
							d3.selectAll(".post").style("border", "2px solid #4d7a89")
						})
					nodes.transition()
						.duration(500)
						.attr("r", "5")

					var comm_tick = (function () {
						var pics = nodes;
						return function () {
							if (typeof pics !== 'undefined')
								pics
								.attr("cx", function (d) {
									return d.x;
								})
								.attr("cy", function (d) {
									return d.y;
								});
						}
					})();

					tick_closures[comm.name] = comm_tick

					var simulation = d3.forceSimulation(comm_pics)
						.force("r", d3.forceRadial(function (d) {
								return comm.r + 10;
							})
							.x(comm.cx)
							.y(comm.cy)
						)
						.on("tick", tick_closures[comm.name]);
					simulations.push(simulation)
				})
		}, 2000);
	});
}

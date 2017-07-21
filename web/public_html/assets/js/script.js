$(document).ready(function () {

    var heatmap_images = {
        1: new Image(),
        2: new Image()
    };

    $(".container").append("<div id='heatmap_image'></div>");
    $("#heatmap_player_1").append(heatmap_images["1"]);
    $("#heatmap_player_2").append(heatmap_images["2"]);





    console.log("Starting SocketIO");
    var socket = io("http://localhost:8080");

    socket.on('connect', function(){
        console.log("Connected to server!")
    });

    socket.on('disconnect', function(){
        console.log("Disconnected from server!")

    });

    socket.on('nn_images', function(data){

    });

    socket.on('heatmap', function(heatmap_data){
        var image = heatmap_images[heatmap_data.player];

        image.src = 'data:image/png;base64,' + heatmap_data.data;
        image.width = "200";
        image.height = "100";


    });

});
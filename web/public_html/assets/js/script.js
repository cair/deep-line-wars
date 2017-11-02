$(document).ready(function () {

    var heatmap_images = {
        1: new Image(),
        2: new Image()
    };



    $(".container").append("<div id='gan_images'></div>");




    console.log("Starting SocketIO");
    var socket = io("http://192.168.160.10:8080"); // TODO

    setInterval(function(){
        $("#.")
    }, 1000);

    socket.on('connect', function(){
        console.log("Connected to server!")
    });

    socket.on('disconnect', function(){
        console.log("Disconnected from server!")

    });




});
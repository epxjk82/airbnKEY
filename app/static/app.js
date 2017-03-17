let get_input_latlng = function() {
    let lat = $("input#lat").val()
    let lng = $("input#lng").val()
    return {'lat': parseFloat(lat),
            'lng': parseFloat(lng)
           }
};

let send_latlng_json = function(latlng) {
    $.ajax({
        url: '/predict',
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        data: JSON.stringify(latlng),
        success: function (data) {
            // display_prediction(data);
            $("img#monthly_inc").toggle();
        }
    });
};

let display_prediction = function(prediction) {
    $("span#predict").html(prediction.prediction)
};

$(document).ready(function() {

    $("button#predict").click(function() {
        let latlng = get_input_latlng();
        send_latlng_json(latlng);

    })

})

function show_div() {
    document.getElementById('form_bag').style.filter = "blur(20px)";
    document.getElementById('loader').style.display = "block";
}


function countdown() {
    var timeleft = document.getElementById('time_field').value
    let downloadTimer = setInterval(function () {
        timeleft--;
        document.getElementById("countdowntimer").textContent = timeleft;
        if (timeleft <= 0)
            clearInterval(downloadTimer);
    }, 1000);
}

function overlay() {
    this.classList.toggle("active");
    let content = this.nextElementSibling;
    if (content.style.display === "block") {
        content.style.display = "none";
    } else {
        content.style.display = "block";
    }
    content.getElementsByClassName()
}

for (let i = 0; i < document.getElementsByClassName("collapsible").length; i++) {
    document.getElementsByClassName("collapsible")[i].addEventListener("click", overlay);
}

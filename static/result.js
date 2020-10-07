function beauty_pythonic_list(id) {
    var txt = document.getElementById(id).innerHTML.replaceAll("[", "").replaceAll("]", "").replaceAll('\'', '');
    document.getElementById(id).innerHTML = "";
    for (let x of txt.split(',')) {
        document.getElementById(id).innerHTML += " " + x
    }
}

function get_elements_quantity(id) {
    return document.getElementById(id).innerHTML.split(" ").length
}


// beauty_pythonic_list("initial_acts")
// beauty_pythonic_list("initial_neurons")
// beauty_pythonic_list("final_acts")
// beauty_pythonic_list("final_neurons")
function find_most_freq(cats) {
    let counts = {}; //We are going to count occurrence of item here
    let compare = 0;  //We are going to compare using stored value
    let mostFrequent;
    for (let i = 0, len = cats.length; i < len; i++) {
        let word = cats[i];

        if (counts[word] === undefined) { //if count[word] doesn't exist
            counts[word] = 1;    //set count[word] value to 1
        } else {                  //if exists
            counts[word] = counts[word] + 1; //increment existing value
        }
        if (counts[word] > compare) {  //counts[word] > 0(first time)
            compare = counts[word];   //set compare to counts[word]
            mostFrequent = cats[i];  //set mostFrequent value
        }
    }
    return mostFrequent
}

function create_table(parent, actid, neurid) {
    var acts = document.getElementById(actid).innerHTML.replaceAll("[", "").replaceAll("]", "").replaceAll('\'', '').split(',')
    var neurs = document.getElementById(neurid).innerHTML.replaceAll("[", "").replaceAll("]", "").replaceAll('\'', '').split(',')
    var table = document.createElement('table'), tr = document.createElement('tr'), cells, i;
    for (i = 0; i < 2; i++) {
        tr.appendChild(document.createElement('td'));
    }
    for (i = 0; i <= acts.length; i++) {
        table.appendChild(tr.cloneNode(true));
    }
    cells = table.getElementsByTagName('td'); // get all of the cells
    let c1 = 0
    let c2 = 0
    let idx0 = 3
    let idx1 = 4
    let sm = 0
    for (i = 0; i < 2 * (acts.length + 1); i++) {
        if (i === 0) {
            cells[i].innerHTML = 'Activations'
        } else if (i === 1) {
            cells[i].innerHTML = 'Neurons Quantity'
        } else if (i === 2) {
        } else if (i === idx0) {
            cells[i].innerHTML = acts[c1].replaceAll(' ', '')
            c1 = c1 + 1;
            idx0 += 3;
        } else if (i === idx1) {
            cells[i].innerHTML = neurs[c2].replaceAll(' ', '')
            sm += parseInt(neurs[c2])
            c2 = c2 + 1;
            idx1 += 3;
        }
    }
    //cells[idx1].innerHTML = sm.toString()
    document.getElementById(parent).appendChild(table);
}

function display_mse_change() {
    let n1 = parseFloat(document.getElementById('initial_mse').innerHTML),
        n2 = parseFloat(document.getElementById('final_mse').innerHTML)
    let diff = (n1 - n2) / n1
    document.getElementById('mse_change').innerHTML = 'Your Final model is ' + diff.toString() + 'better than inital'
}

display_mse_change()
create_table('initial_details_table', "initial_acts", "initial_neurons")
create_table('final_details_table', "final_acts", "final_neurons")
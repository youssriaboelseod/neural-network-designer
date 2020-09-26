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


function find_most_freq(array) {
    if (array.length === 0)
        return null;
    let modeMap = {};
    let maxEl = array[0], maxCount = 1;
    for (let i = 0; i < array.length; i++) {
        let el = array[i];
        if (modeMap[el] == null)
            modeMap[el] = 1;
        else
            modeMap[el]++;
        if (modeMap[el] > maxCount) {
            maxEl = el;
            maxCount = modeMap[el];
        }
    }
    return maxEl;
}

function append_info(parent, n_sum, id, sm, txt) {
    n_sum.id = id
    n_sum.className = id
    n_sum.innerHTML = txt + sm
    document.getElementById(parent).appendChild(n_sum);
}

function generate_table_board_and_fill(i, cells, acts, neurs, parent) {
    let c1 = 0, c2 = 0, idx0 = 3, idx1 = 2, sm = 0
    let cls_quant = 2;
    for (i = 0; i < 2 * (acts.length + 1); i++) {
        if (i === 0) {
            cells[i].innerHTML = 'Neurons Quantity'
        } else if (i === 1) {
            cells[i].innerHTML = 'Activations'
        } else if (i === idx0) {
            cells[i].innerHTML = acts[c1].replaceAll(' ', '')
            c1 = c1 + 1;
            idx0 += cls_quant;
        } else if (i === idx1) {
            cells[i].innerHTML = neurs[c2].replaceAll(' ', '')
            sm += parseInt(neurs[c2])
            c2 = c2 + 1;
            idx1 += cls_quant;
        }
    }
    append_info(parent.slice(0, -6), document.createElement('div'), 'n_sum', sm.toString(), 'All neurons used: ')
    append_info(parent.slice(0, -6), document.createElement('div'), 'l_quant', acts.length - 2, 'Inner layers quantity: ')
    append_info(parent.slice(0, -6), document.createElement('div'), 'most_freq_ac', find_most_freq(acts), 'Most frequent activation: ')
}

function add_environment(acts, neurs) {
    acts.unshift('INPUT')
    acts.push('OUTPUT')
    neurs.unshift('1')
    neurs.push('1')
}

function clear_disturbance(string) {
    return string.replaceAll("[", "").replaceAll("]", "").replaceAll('\'', '').split(',')
}

function create_table(parent, actid, neurid) {
    let acts = clear_disturbance(document.getElementById(actid).innerHTML)
    let neurs = clear_disturbance(document.getElementById(neurid).innerHTML)
    let table = document.createElement('table'), tr = document.createElement('tr'), cells, i;
    add_environment(acts, neurs)
    for (i = 0; i < 2; i++) {
        tr.appendChild(document.createElement('td'));
    }
    for (i = 0; i < acts.length + 1; i++) {
        table.appendChild(tr.cloneNode(true));
    }
    cells = table.getElementsByTagName('td'); // get all of the cells
    generate_table_board_and_fill(i, cells, acts, neurs, parent)
    document.getElementById(parent).appendChild(table);
}

function display_mse_change() {
    let n1 = parseFloat(document.getElementById('initial_mse').innerHTML.split(":")[1]),
        n2 = parseFloat(document.getElementById('final_mse').innerHTML.split(":")[1])
    document.getElementById('final_mse').innerHTML = 'Final MSE:' + n2.toFixed(3)
    document.getElementById('initial_mse').innerHTML = 'Initial MSE: ' + n1.toFixed(3)
    let diff = (n1 - n2) / n1 * 100
    let txt
    if (diff === 0 || isNaN(diff))
        txt = 'Sorry we could not find model better than initial. ' +
            '\n Probably problems: \n ' +
            'working time was not enough- try to give our algorithm some more time. \n' +
            ' initial solution was quite good and algorithm could not find any better solution in that time.'
    else
        txt = 'Your Final model is ' + (diff).toFixed(4) + '% better than initial model.'
    document.getElementById('mse_change').innerText = txt
}

function is_div(el) {
    return el.tagName === 'DIV';
}

function get_value(el) {
    return el.innerHTML.split(':')[1].replace(' ', '')
}

function calc_diff(x1, x2, txt) {
    let par = document.getElementById('diffs')
    let new_it = document.createElement('div')
    new_it.innerHTML = txt + (parseInt(x1) - parseInt(x2)).toString()
    par.appendChild(new_it)

}

function compute_diff(t1, t2,c,txt) {
    let a = document.getElementById(t1);
    let b = document.getElementById(t2);
    let x1 = get_value(a.getElementsByClassName(c))
    let x2 = get_value(b.getElementsByClassName(c))
    calc_diff(x1,x2,txt)
}

create_table('initial_details_table', "initial_acts", "initial_neurons")
create_table('final_details_table', "final_acts", "final_neurons")
display_mse_change()
compute_diff('initial_details', 'final_details','n_sum','All neurons used difference: ')
compute_diff('initial_details', 'final_details','l_quant','Inner Layers quantity diff: ')


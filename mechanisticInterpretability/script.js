const WEIGHTS = `4
5,4,4,1
0.3665154655529764,0.26940883787890985,-3.6170572100873137,2.2592167974727113,0.7375309723139867,1.7311501429936333,1.1556809851523688,0.2699647852259367,-0.2629985531712496,-0.4539206437470659,-0.8515832100485103,-1.1671459246634823,0.46874961944242727,-1.3744188327978915,-0.18842507360336472,-0.06144223631459124,1.4102878025107457,-0.7616779420999777,-1.63694297153991,1.1578162573217081,-0.44534500298907365,0.898817056495226,-0.6382074719821649,-1.2188772513087136
-0.4366593985305829,-0.16647477745046746,1.924839377072285,-0.0977926165871548,1.5857424376010605,-0.3668600476660198,-0.8990566755740517,-0.5873416871541978,-1.893612513701998,-1.0606940868305639,-0.4677780065513302,1.3676293993103492,-0.46999596393141146,-2.582452711369717,1.8266093304655988,0.013445311346923776,0.5708864328501712,0.19680073352558863,0.6809822144678884,1.9006905942295529
-0.588682070747429,0.31455834788995524,0.9212212112415777,0.3342437819604826,0.10549227424479435`

let layers = []
let weights = []
let biases = []

const makeArray = (x,y) => {
    var arr = new Array(x);
    for (var i = 0; i < arr.length; i++) {
        arr[i] = new Array(y);
    }
    return arr
}

const setup = () => {
    let lines = WEIGHTS.split("\n")
    console.log(lines)

    layers = lines[1].split(",").map(x => Number(x))
    for (let i = 1; i < layers.length; i++) {
        let data = lines[i + 1].split(",").map(x => Number(x))

        let weights_array = makeArray(layers[i-1], layers[i])
        let index = 0
        for (let x = 0; x < layers[i-1]; x++) {
            for (let y = 0; y < layers[i]; y++) {
                weights_array[x][y] = data[index]
                index += 1
            }
        }
        
        let bias_array = Array(layers[i])
        for (let b = 0; b < layers[i]; b++) {
            bias_array[b] = data[index]
            index += 1            
        }

        weights.push(weights_array)
        biases.push(bias_array)
    }

    const rerender = () => {
        inputs = [
            Number(document.getElementById("input0").value),
            Number(document.getElementById("input1").value),
            Number(document.getElementById("input2").value),
            Number(document.getElementById("input3").value),
            Number(document.getElementById("input4").value)
        ]
        drawNetwork(inputs)
    }

    //Setup HTML
    document.getElementById("input0").oninput = rerender
    document.getElementById("input1").oninput = rerender
    document.getElementById("input2").oninput = rerender
    document.getElementById("input3").oninput = rerender
    document.getElementById("input4").oninput = rerender
}

const drawNetwork = (inputs) => {
    ctx = document.getElementById("canvas").getContext("2d")

    // Compute
    let nodeActivations = [inputs]
    let outputs
    for (let i = 0; i < weights.length; i++) {
        outputs = Array(layers[i + 1]).fill(0)

        // Apply mat multiply
        for (let j = 0; j < weights[i].length; j++) {
            for (let k = 0; k < weights[i][j].length; k++) {
                const weight = weights[i][j][k]
                outputs[k] += weight * inputs[j]    
            }
        }

        // Apply biases
        for (let j = 0; j < biases[i].length; j++) {
            outputs[j] += biases[i][j]    
        }

        // Apply tanh
        for (let j = 0; j < biases[i].length; j++) {
            outputs[j] = Math.tanh(outputs[j])   
        }

        nodeActivations.push(outputs)
        inputs = outputs
    }

    console.log(nodeActivations)

    // Draw nodes
    for (let i = 0; i < layers.length; i++) {
        const layerCount = layers[i];
        for (let j = 0; j < layerCount; j++) {
            const HEIGHT = 600
            y = (HEIGHT / (layerCount + 1)) * (j + 1)

            let fraction = (i == 0 && j != 4 ? 0.2 : 1.0)

            ctx.beginPath();
            ctx.fillStyle = `rgb(${255 * fraction * nodeActivations[i][j]},0,0)`;
            ctx.arc(20 + (i * 250), y, 10, 0, 2 * Math.PI, false)
            ctx.fill()
        }
    }

    // Draw weights
}


setup()
drawNetwork([1,2,3,4,0])
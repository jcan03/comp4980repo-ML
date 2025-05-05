// initialize variables
let timeLeft = 60;
let timer = document.getElementById("timer");
let textbox = document.getElementById("textbox");
let submitButton = document.getElementById("submit");
let countdown;
let started = false;
let keystrokeData = [];
let pressTimes = {};
let sessionStart = null;
let num = 0;

// calculate Typing Speed (Keystrokes per minute)
function calculateTypingSpeed(data) {
    if (data.length === 0) return 0;

    // exclude backspaces from the total keystroke count
    let totalKeystrokes = data.length;
    if (totalKeystrokes === 0) return 0;

    return totalKeystrokes;
}

// calculate flight time variability 
function calculateFlightTimeVariability(data) {
    let ilValues = [];
    for (let i = 1; i < data.length; i++) {
        let il = parseFloat(data[i].relativePressTime) - parseFloat(data[i - 1].relativeReleaseTime);
        if (il > 0) {  // only include positive IL values
            ilValues.push(Math.log(il));  // log transformation here for standard deviation
        }
    }

    if (ilValues.length <= 1) return 0;

    let mean = ilValues.reduce((a, b) => a + b, 0) / ilValues.length;
    let variance = ilValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (ilValues.length - 1);
    return Math.sqrt(variance);
}

// calculate HLSD (Hold Latency Standard Deviation)
function calculateHLSD(data) {
    let hlValues = data
        .map(entry => parseFloat(entry.sTap))
        .filter(hl => hl > 0);  // only positive values, in intiial dataset there were issues, so want to be safe here

    if (hlValues.length <= 1) return 0;

    let logHLValues = hlValues.map(hl => Math.log(hl));  // log transformation for standard deviation
    let mean = logHLValues.reduce((a, b) => a + b, 0) / logHLValues.length;
    let variance = logHLValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (logHLValues.length - 1);
    return Math.sqrt(variance);
}

// calculate RLSD (Release Latency Standard Deviation)
function calculateRLSD(data) {
    let rlValues = [];
    for (let i = 1; i < data.length; i++) {
        let rl = parseFloat(data[i].relativeReleaseTime) - parseFloat(data[i - 1].relativeReleaseTime);
        if (rl > 0) {  // only positive values, in intiial dataset there were issues, so want to be safe here
            rlValues.push(Math.log(rl));  //log transformation for standard deviation
        }
    }

    if (rlValues.length <= 1) return 0;

    let mean = rlValues.reduce((a, b) => a + b, 0) / rlValues.length;
    let variance = rlValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (rlValues.length - 1);
    return Math.sqrt(variance);
}

// calculate Average Hold Duration
function calculateAvgHoldDuration(data) {
    if (data.length === 0) return 0;
    let totalHoldDuration = data.reduce((sum, entry) => sum + parseFloat(entry.sTap), 0); 
    return totalHoldDuration / data.length; // how long all keys were pressed for divided by num of keys pressed
}

// calculate Backspace Rate
function calculateBackspaceRate(data) {
    if (data.length === 0) return 0;
    let backspaceCount = data.filter(entry => entry.key === "Backspace").length;

    if (backspaceCount === 0) return 0.000000001; // if 0, return very low number because 0 scews data (matches dataset)

    else return backspaceCount / data.length;
}

// start countdown when typing begins (1 second between typing and timer going down)
textbox.addEventListener("input", () => {
    if (!started) {
        started = true;
        sessionStart = Date.now();
        countdown = setInterval(() => {
            timeLeft--;
            timer.textContent = timeLeft;
            if (timeLeft <= 0) {
                clearInterval(countdown);
                textbox.disabled = true;
                submitButton.disabled = false;
            }
        }, 1000);
    }
});

// capture key presses (keydown)
textbox.addEventListener("keydown", (event) => {
    if (!pressTimes[event.key]) {
        pressTimes[event.key] = Date.now();
    }
});

// capture key releases (keyup)
textbox.addEventListener("keyup", (event) => {
    let pressTime = pressTimes[event.key];
    if (pressTime) {
        let releaseTime = Date.now();
        let holdTime = (releaseTime - pressTime) / 1000;
        let relativePressTime = (pressTime - sessionStart) / 1000;
        let relativeReleaseTime = (releaseTime - sessionStart) / 1000;

        // push all of the extracted features, defaulting to 0 for the most part
        keystrokeData.push({
            num: 0,
            pID: 0,
            file_2: '',
            gt: '',
            updrs108: 0,
            afTap: 0,
            sTap: holdTime.toFixed(4),
            nqScore: 0,
            typingSpeed: 0,
            dataset: '',
            file_1: '',
            avgHoldDuration: 0,
            backspaceRate: 0,
            holdingLatencySD: 0,
            releaseLatencySD: 0,
            flightTimeVariability: 0,
            relativePressTime: relativePressTime.toFixed(4),
            relativeReleaseTime: relativeReleaseTime.toFixed(4),
            key: event.key
        });

        delete pressTimes[event.key]; // delete press times because we do not want these in the saved csv
    }
});

// save captured data as CSV and submit to backend with actual calculated values
async function submitText() {
    console.log("Submit button clicked!");

    let typingSpeed = calculateTypingSpeed(keystrokeData);
    let flightTimeVariability= calculateFlightTimeVariability(keystrokeData);
    let holdingLatencySD = calculateHLSD(keystrokeData);
    let releaseLatencySD = calculateRLSD(keystrokeData);
    let avgHoldDuration = calculateAvgHoldDuration(keystrokeData);
    let backspaceRate = calculateBackspaceRate(keystrokeData);

    // update calculated values in the data
    keystrokeData.forEach(entry => {
        entry.typingSpeed = typingSpeed.toFixed(4);
        entry.flightTimeVariability = flightTimeVariability.toFixed(9);
        entry.holdingLatencySD = holdingLatencySD.toFixed(9);
        entry.releaseLatencySD = releaseLatencySD.toFixed(9);
        entry.avgHoldDuration = avgHoldDuration.toFixed(9);
        entry.backspaceRate = backspaceRate.toFixed(9);
        delete entry.key;  // remove temporary key attribute before saving
    });

    const cleanData = keystrokeData.map(({ relativePressTime, relativeReleaseTime, ...rest }) => rest);
    console.log("Data to be sent:", cleanData);

    try {
        const response = await fetch('/predict', { // try the predict function, if successful give Y/N response, if error, give error msg
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ keystrokes: cleanData }),
        });

        const result = await response.json();
        console.log("Prediction result:", result);

        if (response.ok) {
            alert('Prediction: ' + result.prediction);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        alert('Failed to send data to the server.');
        console.error("Submission error:", error);
    }

    keystrokeData = [];
    num = 0;
    started = false;
}

submitButton.addEventListener("click", submitText); // once clicked, predict function should be called using all the tracked/saved keystroke data

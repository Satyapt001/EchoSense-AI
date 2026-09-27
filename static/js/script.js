let selectedMode = "animal";

let mediaRecorder = null;
let audioChunks = [];
let recTimerInterval = null;
let recSeconds = 0;
let isRecording = false;


// ==================================================
// DOM ELEMENTS
// ==================================================

const audioFile = document.getElementById("audioFile");
const dropZone = document.getElementById("dropZone");
const fileCard = document.getElementById("fileCard");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const audioPreview = document.getElementById("audioPreview");
const predictBtn = document.getElementById("predictBtn");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const confidenceBar = document.getElementById("confidenceBar");
const error = document.getElementById("error");
const errorMessage = document.getElementById("errorMessage");

const micRecordBtn =
    document.getElementById("micRecordBtn");

const micBtnText =
    document.getElementById("micBtnText");

const recStatus =
    document.getElementById("recStatus");

const recTimer =
    document.getElementById("recTimer");

const probabilitiesSection =
    document.getElementById("probabilitiesSection");

const probabilitiesList =
    document.getElementById("probabilitiesList");


// ==================================================
// MODE
// ==================================================

function selectMode(mode) {

    if (isRecording) {
        stopRecording();
    }

    selectedMode = mode;

    document
        .querySelectorAll(".mode-btn")
        .forEach(button => {

            button.classList.toggle(
                "active",
                button.dataset.mode === mode
            );

        });


    // Microphone is only available for speakers
    if (micRecordBtn) {

        micRecordBtn.classList.toggle(
            "hidden",
            mode !== "speaker"
        );

    }


    // Update UI text
    const dropTitle =
        document.querySelector(".drop-zone h4") ||
        document.querySelector(".drop-zone h3");

    const dropDescription =
        document.querySelector(".drop-zone p");

    if (mode === "speaker") {

        if (dropTitle) {
            dropTitle.textContent =
                "Upload or record your voice";
        }

        if (dropDescription) {
            dropDescription.textContent =
                "Upload an audio file or use your microphone";
        }

    } else {

        if (dropTitle) {
            dropTitle.textContent =
                "Drop your animal audio here";
        }

        if (dropDescription) {
            dropDescription.textContent =
                "WAV, MP3, OGG, or FLAC up to 25MB";
        }

    }


    clearFile();
    clearResult();

    lucide.createIcons();
}


// ==================================================
// FILE SELECTION
// ==================================================

audioFile.addEventListener(
    "change",
    function () {

        if (this.files.length) {
            handleFile(this.files[0]);
        }

    }
);


// ==================================================
// DRAG & DROP
// ==================================================

dropZone.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        dropZone.classList.add(
            "dragging"
        );

    }
);


dropZone.addEventListener(
    "dragleave",
    function () {

        dropZone.classList.remove(
            "dragging"
        );

    }
);


dropZone.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        dropZone.classList.remove(
            "dragging"
        );

        const files =
            event.dataTransfer.files;

        if (files.length) {
            handleFile(files[0]);
        }

    }
);


// ==================================================
// HANDLE FILE
// ==================================================

function handleFile(file) {

    if (
        !file.type.startsWith("audio/") &&
        !file.name.match(
            /\.(wav|mp3|ogg|flac|m4a|webm)$/i
        )
    ) {

        showError(
            "Please select a valid audio file."
        );

        return;
    }


    const maxSize =
        25 * 1024 * 1024;

    if (file.size > maxSize) {

        showError(
            "Audio file must be smaller than 25 MB."
        );

        return;
    }


    clearResult();
    hideError();


    fileName.textContent =
        file.name;

    fileSize.textContent =
        formatFileSize(file.size);

    fileCard.classList.remove(
        "hidden"
    );


    const objectURL =
        URL.createObjectURL(file);

    audioPreview.src =
        objectURL;

    audioPreview.classList.add(
        "visible"
    );


    // Keep file in input
    const dataTransfer =
        new DataTransfer();

    dataTransfer.items.add(file);

    audioFile.files =
        dataTransfer.files;


    lucide.createIcons();
}


// ==================================================
// FILE SIZE
// ==================================================

function formatFileSize(bytes) {

    if (bytes < 1024) {
        return bytes + " B";
    }

    if (bytes < 1024 * 1024) {

        return (
            bytes / 1024
        ).toFixed(1) + " KB";

    }

    return (
        bytes / (1024 * 1024)
    ).toFixed(2) + " MB";
}


// ==================================================
// REMOVE FILE
// ==================================================

function removeFile() {
    clearFile();
    clearResult();
}


function clearFile() {

    audioFile.value = "";

    fileCard.classList.add(
        "hidden"
    );

    audioPreview.pause();

    audioPreview.removeAttribute(
        "src"
    );

    audioPreview.classList.remove(
        "visible"
    );
}


// ==================================================
// MICROPHONE RECORDING
// ==================================================

async function toggleMicRecord() {

    if (selectedMode !== "speaker") {

        showError(
            "Microphone recording is available only for speaker recognition."
        );

        return;
    }


    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}


// ==================================================
// START RECORDING
// ==================================================

async function startRecording() {

    if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
    ) {

        showError(
            "Microphone recording is not supported by this browser."
        );

        return;
    }


    try {

        hideError();
        clearFile();
        clearResult();


        const stream =
            await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });


        audioChunks = [];


        let mimeType = "";

        if (
            MediaRecorder.isTypeSupported(
                "audio/webm;codecs=opus"
            )
        ) {

            mimeType =
                "audio/webm;codecs=opus";

        } else if (
            MediaRecorder.isTypeSupported(
                "audio/webm"
            )
        ) {

            mimeType =
                "audio/webm";

        }


        mediaRecorder = mimeType
            ? new MediaRecorder(
                stream,
                { mimeType }
            )
            : new MediaRecorder(stream);


        mediaRecorder.ondataavailable =
            event => {

                if (event.data.size > 0) {
                    audioChunks.push(
                        event.data
                    );
                }

            };


        mediaRecorder.onstop =
            async () => {

                const finalMimeType =
                    mediaRecorder.mimeType ||
                    "audio/webm";

                const extension =
                    finalMimeType.includes(
                        "webm"
                    )
                        ? "webm"
                        : "ogg";


                const audioBlob =
                    new Blob(
                        audioChunks,
                        {
                            type:
                                finalMimeType
                        }
                    );


                const recordedFile =
                    new File(
                        [audioBlob],
                        `recorded_voice_${Date.now()}.${extension}`,
                        {
                            type:
                                finalMimeType
                        }
                    );


                // Stop microphone
                stream
                    .getTracks()
                    .forEach(track => {
                        track.stop();
                    });


                // Put recording into normal file flow
                handleFile(
                    recordedFile
                );


                // Automatically analyze recording
                await predictAudio();
            };


        mediaRecorder.start();

        isRecording = true;

        micRecordBtn.classList.add(
            "recording"
        );

        micBtnText.textContent =
            "Recording...";


        recStatus.classList.remove(
            "hidden"
        );


        recSeconds = 0;

        updateRecTimerDisplay();


        recTimerInterval =
            setInterval(() => {

                recSeconds++;

                updateRecTimerDisplay();

            }, 1000);


    } catch (err) {

        console.error(
            "Microphone error:",
            err
        );

        showError(
            "Microphone permission was denied or the microphone could not be accessed."
        );
    }
}


// ==================================================
// STOP RECORDING
// ==================================================

function stopRecording() {

    if (
        mediaRecorder &&
        isRecording
    ) {

        mediaRecorder.stop();

        isRecording = false;


        micRecordBtn.classList.remove(
            "recording"
        );

        micBtnText.textContent =
            "Record Mic";


        recStatus.classList.add(
            "hidden"
        );


        if (recTimerInterval) {

            clearInterval(
                recTimerInterval
            );

            recTimerInterval = null;
        }
    }
}


// ==================================================
// RECORDING TIMER
// ==================================================

function updateRecTimerDisplay() {

    const mins =
        String(
            Math.floor(
                recSeconds / 60
            )
        ).padStart(2, "0");


    const secs =
        String(
            recSeconds % 60
        ).padStart(2, "0");


    recTimer.textContent =
        `${mins}:${secs}`;
}


// ==================================================
// PREDICTION
// ==================================================

async function predictAudio() {

    if (!audioFile.files.length) {

        showError(
            "Please upload an audio file or record from microphone first."
        );

        return;
    }


    const formData =
        new FormData();

    formData.append(
        "audio",
        audioFile.files[0]
    );

    formData.append(
        "mode",
        selectedMode
    );


    predictBtn.disabled = true;

    // Hide the placeholder while analyzing
    if (resultPlaceholder) {
        resultPlaceholder.classList.add("hidden");
    }

    loading.classList.remove(
        "hidden"
    );

    result.classList.add(
        "hidden"
    );

    hideError();


    try {

        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error ||
                "Prediction failed."
            );
        }


        if (!data.success) {

            throw new Error(
                data.error ||
                "Prediction failed."
            );
        }


        showResult(
            data.prediction,
            data.confidence,
            data.probabilities
        );


    } catch (err) {

        console.error(
            "Prediction error:",
            err
        );

        showError(
            err.message ||
            "Something went wrong."
        );

        // Bring back placeholder if prediction failed
        if (resultPlaceholder) {
            resultPlaceholder.classList.remove("hidden");
        }

    } finally {

        predictBtn.disabled = false;

        loading.classList.add(
            "hidden"
        );
    }
}


// ==================================================
// SHOW RESULT
// ==================================================

function showResult(
    predictedClass,
    confidenceValue,
    probabilities
) {

    // Ensure the "Awaiting Audio" section stays hidden
    if (resultPlaceholder) {
        resultPlaceholder.classList.add("hidden");
    }

    prediction.textContent =
        predictedClass;


    confidence.textContent =
        confidenceValue + "%";


    result.classList.remove(
        "hidden"
    );


    setTimeout(() => {

        confidenceBar.style.width =
            confidenceValue + "%";

    }, 100);


    // Probability breakdown
    if (
        probabilities &&
        Object.keys(probabilities).length > 0
    ) {

        probabilitiesList.innerHTML =
            "";


        const sortedProbs =
            Object.entries(
                probabilities
            ).sort(
                (a, b) => b[1] - a[1]
            );


        sortedProbs.forEach(
            ([className, score]) => {

                const item =
                    document.createElement(
                        "div"
                    );

                item.className =
                    "prob-item";


                const row =
                    document.createElement(
                        "div"
                    );

                row.className =
                    "prob-row";


                const label =
                    document.createElement(
                        "span"
                    );

                label.className =
                    "prob-class";

                label.textContent =
                    className.replace(
                        /_/g,
                        " "
                    );


                const val =
                    document.createElement(
                        "span"
                    );

                val.className =
                    "prob-value";

                val.textContent =
                    score + "%";


                row.appendChild(label);
                row.appendChild(val);


                const barBg =
                    document.createElement(
                        "div"
                    );

                barBg.className =
                    "prob-bar-bg";


                const barFill =
                    document.createElement(
                        "div"
                    );

                barFill.className =
                    "prob-bar-fill";

                barFill.style.width =
                    "0%";


                barBg.appendChild(
                    barFill
                );

                item.appendChild(
                    row
                );

                item.appendChild(
                    barBg
                );


                probabilitiesList.appendChild(
                    item
                );


                setTimeout(() => {

                    barFill.style.width =
                        score + "%";

                }, 120);
            }
        );


        probabilitiesSection.classList.remove(
            "hidden"
        );

    } else {

        probabilitiesSection.classList.add(
            "hidden"
        );
    }


    result.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });


    lucide.createIcons();
}


// ==================================================
// ERROR
// ==================================================

function showError(message) {

    errorMessage.textContent =
        message;

    error.classList.remove(
        "hidden"
    );

    lucide.createIcons();
}


function hideError() {

    error.classList.add(
        "hidden"
    );
}


// ==================================================
// CLEAR RESULT
// ==================================================

function clearResult() {

    result.classList.add(
        "hidden"
    );

    // Show the "Awaiting Audio" placeholder again
    if (resultPlaceholder) {
        resultPlaceholder.classList.remove("hidden");
    }

    hideError();

    confidenceBar.style.width =
        "0%";


    if (probabilitiesSection) {

        probabilitiesSection.classList.add(
            "hidden"
        );
    }
}


// ==================================================
// INITIALIZE
// ==================================================

selectMode("animal");

lucide.createIcons();
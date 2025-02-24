import { Client } from "https://esm.sh/@gradio/client";

async function uploadImages(event) {
    event.preventDefault();

    let referenceImage = document.getElementById('referenceImage').files[0];
    let inputImage = document.getElementById('inputImage').files[0];

    if (!referenceImage || !inputImage) {
        alert("Please select both images before uploading.");
        return;
    }

    document.getElementById("loadingMessage").style.display = "block";
    document.getElementById("resultContainer").style.display = "none";

    try {
        const client = await Client.connect("Vestina/face-recognition-api");  // Correct HF API
        const result = await client.predict("/predict", {
            reference_image: referenceImage,
            input_image: inputImage
        });

        console.log("API Response:", result.data);

        if (result.data && result.data.length > 1 && result.data[1].url) {
            document.getElementById("outputImage").src = result.data[1].url;
            document.getElementById("outputImage").style.display = "block";
            document.getElementById("resultContainer").style.display = "block";
        } else {
            alert("Error: Could not retrieve processed image.");
        }

    } catch (error) {
        console.error("Error:", error);
        alert("Something went wrong! Please check your API.");
    } finally {
        document.getElementById("loadingMessage").style.display = "none";
    }
}

document.getElementById("faceRecognitionForm").addEventListener("submit", uploadImages);

import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";

const CameraStreamWithDetection = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [streaming, setStreaming] = useState(false);
    const [customModel, setCustomModel] = useState(null);
    const [result, setResult] = useState("");

    // Load custom TensorFlow.js model
    useEffect(() => {
        const loadModel = async () => {
            try {
                const model = await tf.loadGraphModel("/model2/model.json");
                setCustomModel(model);
                console.log("Model loaded successfully");
                // After loading the model, inspect its inputs
                // console.log("Model inputs:", model.inputs);
                // console.log("Expected input shape:", model.inputs[0].shape);
            } catch (err) {
                console.error("Failed to load model:", err);
            }
        };
        loadModel();
    }, []);

    // Start camera
    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                setStreaming(true);
            }
        } catch (err) {
            console.error("Error accessing camera:", err);
        }
    };

    // Stop camera
    const stopCamera = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            const stream = videoRef.current.srcObject;
            stream.getTracks().forEach((track) => track.stop());
            videoRef.current.srcObject = null;
        }
        setStreaming(false);
    };

    // Analyze frames every 3 seconds
    useEffect(() => {
        if (!streaming || !customModel) return;

        const analyzeFrame = async () => {
            if (!videoRef.current || !canvasRef.current) return;

            const MODEL_INPUT_SIZE = 224; // Adjust if your model expects different dimensions
            const canvas = canvasRef.current;
            const ctx = canvas.getContext("2d");

            // Set canvas to model's expected size
            canvas.width = MODEL_INPUT_SIZE;
            canvas.height = MODEL_INPUT_SIZE;
            ctx.drawImage(videoRef.current, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

            try {
                // 1. Get model's expected input details
                const inputName = customModel.inputs[0].name;
                // console.log(`Model input: ${inputName}, Shape: ${customModel.inputs[0].shape}`);

                // 2. Create input tensor with channels-first ordering
                const inputTensor = tf.tidy(() => {
                    // Convert to tensor and normalize
                    const tensor = tf.browser.fromPixels(canvas) // Shape: [224, 224, 3]
                        .toFloat()
                        .div(255.0);

                    // Transpose to channels-first format [3, 224, 224]
                    const transposedTensor = tensor.transpose([2, 0, 1]);
                    // Add batch dimension [1, 3, 224, 224]
                    return transposedTensor.expandDims(0);
                });

                // console.log("Final input tensor shape:", inputTensor.shape);

                // 3. Execute model with proper input format
                const outputs = await customModel.executeAsync({
                    [inputName]: inputTensor
                });

                // 4. Process outputs
                // const results = await outputs.data();
                // console.log("Predictions:", results);

                // 4. Process outputs
                // const probabilities = await tf.softmax(outputs).data(); // Convert logits to probabilities
                // console.log("Predictions:", {
                //     Normal: probabilities[0], // First probability as Normal
                //     NSFW: probabilities[1]    // Second probability as NSFW
                // });

                const probabilities = await tf.softmax(outputs).data();
                // const prediction = probabilities[0] > probabilities[1] ? "Normal" : "NSFW";
                // const maxProbability = Math.max(probabilities[0], probabilities[1]);
                const prediction = probabilities[1] >= 0.7 ? "NSFW" : "Normal";
                const probability = prediction === "NSFW" ? probabilities[1] : probabilities[0];
                const data = `Predicted: ${prediction} (${(probability * 100).toFixed(2)}%)`
                setResult(data);
                // Log only the class with the highest probability
                // console.log(`Predicted Class: ${prediction}, Probability: ${(maxProbability * 100).toFixed(2)}%`);
                console.log(`Predicted: ${prediction} (${(probability * 100).toFixed(2)}%)`);
                // 5. Cleanup
                tf.dispose([inputTensor, outputs]);
            } catch (err) {
                console.error("Prediction failed:", err);

                // Advanced debugging
                if (customModel?.layers) {
                    const firstConvLayer = customModel.layers.find(l => l.className.includes('Conv'));
                    if (firstConvLayer) {
                        console.log("First conv layer filters shape:",
                            firstConvLayer.weights[0].shape); // Should be [H, W, 3, N]
                    }
                }
            }
        };

        const interval = setInterval(analyzeFrame, 3000);
        return () => clearInterval(interval);
    }, [streaming, customModel]);

    return (
        <div className="p-4 text-center">
            <h1 className="text-2xl font-bold mb-4">Live Stream + Custom Detection</h1>

            <div className="flex justify-center mb-4">
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    className="w-full max-w-md rounded-lg shadow-lg border border-gray-300"
                />
            </div>

            <div className="space-x-2">
                {!streaming ? (
                    <button
                        onClick={startCamera}
                        className="px-4 py-2 bg-green-600 text-white rounded shadow hover:bg-green-700"
                    >
                        Start Camera
                    </button>
                ) : (
                    <button
                        onClick={stopCamera}
                        className="px-4 py-2 bg-red-600 text-white rounded shadow hover:bg-red-700"
                    >
                        Stop Camera
                    </button>
                )}
            </div>

            <div className="mt-4">
                <p>Camera Status: {streaming ? "Streaming" : "Stop"}</p>
            </div>
            <div className="mt-4">
                <p>{streaming ? result : "Waiting..."}</p>
            </div>

            {/* Hidden canvas for frame capture */}
            <canvas
                ref={canvasRef}
                width={224}
                height={224}
                style={{ display: "none" }}
            />
        </div>
    );
};

export default CameraStreamWithDetection;

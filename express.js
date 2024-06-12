const express = require('express');
const cv = require('opencv4nodejs');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const { PythonShell } = require('python-shell');

const app = express();

// Set up body parser middleware
app.use(bodyParser.json());

// Define paths
const modelPath = '/path/to/your/model'; // Update with your model path
const imagesFolder = '/path/to/your/images'; // Update with your images folder path

// Load the trained model
const model = loadModel(modelPath);

// Load metadata if needed
const metadataPath = '/path/to/your/metadata.csv'; // Update with your metadata path
const metadata = loadMetadata(metadataPath);

// Define symptoms for each class
const symptoms = {
  'akiec': 'Actinic keratoses and intraepithelial carcinoma / Bowen\'s disease',
  'bcc': 'Basal cell carcinoma',
  'bkl': 'Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)',
  'df': 'Dermatofibroma',
  'mel': 'Melanoma',
  'nv': 'Melanocytic nevi',
  'vasc': 'Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)'
};

// Define routes
app.post('/predict', async (req, res) => {
  try {
    const base64Image = req.body.image;
    const imageBuffer = Buffer.from(base64Image, 'base64');
    const tempImagePath = path.join(__dirname, 'tempImage.jpg');
    
    // Save image buffer to temporary file
    fs.writeFileSync(tempImagePath, imageBuffer);

    // Read image using OpenCV
    const img = cv.imread(tempImagePath);

    // Preprocess image (resize to 128x128)
    const resizedImg = img.resize(128, 128);

    // Convert image to array
    const imgArray = cv.cvtColor(resizedImg, cv.COLOR_BGR2RGB).getDataAsArray();

    // Make prediction
    const prediction = model.predict(imgArray);
    const predictedLabelIndex = prediction.argmax();

    // Get predicted label and corresponding symptom
    const predictedLabel = Object.keys(symptoms)[predictedLabelIndex];
    const predictedSymptom = symptoms[predictedLabel];

    // Send response
    res.json({
      predictedLabel,
      predictedSymptom
    });

    // Delete temporary image file
    fs.unlinkSync(tempImagePath);
  } catch (error) {
    console.error('Error:', error.message);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// Load the model
function loadModel(modelPath) {
  // Add code to load your TensorFlow.js model
  // Example: return tf.loadLayersModel(`file://${modelPath}`);
}

// Load metadata
function loadMetadata(metadataPath) {
  // Add code to load your metadata
}

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
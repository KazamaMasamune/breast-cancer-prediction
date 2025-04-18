document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const resultDiv = document.getElementById('result');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (response.ok) {
            resultDiv.className = 'result success';
            resultDiv.innerHTML = `Prediction: ${data.prediction} (Confidence: ${data.confidence})`;
        } else {
            resultDiv.className = 'result error';
            resultDiv.innerHTML = `Error: ${data.error}`;
        }
    } catch (error) {
        resultDiv.className = 'result error';
        resultDiv.innerHTML = `Error: ${error.message}`;
    }
});
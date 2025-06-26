document.addEventListener('DOMContentLoaded', function () {
    if (document.getElementById('loginForm')) {
        const loginForm = document.getElementById('loginForm');
        loginForm.addEventListener('submit', function (e) {
            e.preventDefault();
            const email = document.getElementById('email').value;
            sessionStorage.setItem('isLoggedIn', 'true');
            sessionStorage.setItem('userEmail', email);
            window.location.href = 'dashboard.html';
        });
    }

    if (document.getElementById('detectionOption')) {
        const isLoggedIn = sessionStorage.getItem('isLoggedIn');
        if (!isLoggedIn) {
            window.location.href = 'index.html';
            return;
        }

        document.getElementById('logoutBtn').addEventListener('click', function () {
            sessionStorage.removeItem('isLoggedIn');
            sessionStorage.removeItem('userEmail');
            window.location.href = 'index.html';
        });

        const detectionOption = document.getElementById('detectionOption');
        const precautionsOption = document.getElementById('precautionsOption');
        const detectionSection = document.getElementById('detectionSection');
        const precautionsSection = document.getElementById('precautionsSection');

        detectionOption.addEventListener('click', function () {
            detectionSection.classList.remove('hidden');
            precautionsSection.classList.add('hidden');
        });

        precautionsOption.addEventListener('click', function () {
            precautionsSection.classList.remove('hidden');
            detectionSection.classList.add('hidden');
        });

        document.getElementById('backToOptions').addEventListener('click', function () {
            detectionSection.classList.add('hidden');
        });

        document.getElementById('backToOptions2').addEventListener('click', function () {
            precautionsSection.classList.add('hidden');
        });

        const imageUpload = document.getElementById('imageUpload');
        const uploadBtn = document.getElementById('uploadBtn');
        const uploadedImage = document.getElementById('uploadedImage');
        const resultText = document.getElementById('resultText');
        const resultsContainer = document.getElementById('resultsContainer');
        const progressContainer = document.getElementById('progressContainer');
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');

        let selectedFile = null;

        imageUpload.addEventListener('change', function (e) {
            if (e.target.files.length > 0) {
                selectedFile = e.target.files[0];
                const reader = new FileReader();
                reader.onload = function (event) {
                    uploadedImage.src = event.target.result;
                };
                reader.readAsDataURL(selectedFile);
                uploadBtn.disabled = false;
                resultText.textContent = '';
            }
        });

        uploadBtn.addEventListener('click', async function () {
            if (!selectedFile) return;

            uploadBtn.disabled = true;
            progressContainer.classList.remove('hidden');
            progressBar.style.width = '0%';
            progressText.textContent = '0%';
            resultText.textContent = 'Uploading image...';
            resultsContainer.classList.remove('hidden');

            const accessKey = "AWS---Access---key";
            const secretKey = "AWS---Secret---Key";
            const region = "us-east-1";
            const bucket = "deepfake222";

            AWS.config.update({
                region: region,
                credentials: new AWS.Credentials(accessKey, secretKey)
            });

            const s3 = new AWS.S3({ apiVersion: '2006-03-01' });

            try {
                const timestamp = Date.now();
                const sanitizedFilename = selectedFile.name.replace(/\s+/g, '_');
                const uploadKey = `uploads/${timestamp}_${sanitizedFilename}`;

                const upload = new AWS.S3.ManagedUpload({
                    params: {
                        Bucket: bucket,
                        Key: uploadKey,
                        Body: selectedFile,
                        ContentType: selectedFile.type,
                        ACL: 'public-read'
                    }
                });

                upload.on('httpUploadProgress', (progress) => {
                    const percent = Math.round((progress.loaded / progress.total) * 100);
                    progressBar.style.width = `${percent}%`;
                    progressText.textContent = `${percent}%`;
                });

                await upload.promise();

                resultText.textContent = 'Upload successful! Checking for results...';

                // Keep full filename including timestamp
                const uploadedFileName = uploadKey.replace(/^uploads\//, '');
                const expectedJsonKey = `results/${uploadedFileName}.json`;

                const maxRetries = 20;
                let retries = 0;

                const checkForResult = async () => {
                    try {
                        const listedObjects = await s3.listObjectsV2({
                            Bucket: bucket,
                            Prefix: 'results/'
                        }).promise();

                        const jsonKeys = listedObjects.Contents
                            .map(obj => obj.Key)
                            .filter(key => key.endsWith('.json'));

                        if (jsonKeys.includes(expectedJsonKey)) {
                            const jsonObject = await s3.getObject({
                                Bucket: bucket,
                                Key: expectedJsonKey
                            }).promise();

                            const jsonStr = new TextDecoder('utf-8').decode(jsonObject.Body);
                            const resultJson = JSON.parse(jsonStr);

                            if (resultJson.label && resultJson.confidence !== undefined) {
    resultText.textContent = `Prediction: ${resultJson.label.toUpperCase()} (Confidence: ${resultJson.confidence.toFixed(2)}%)`;
} else {
    resultText.textContent = 'Result JSON found but missing expected fields.';
}

                            progressContainer.classList.add('hidden');
                            return;
                        } else {
                            if (retries < maxRetries) {
                                retries++;
                                resultText.textContent = `Waiting for result... Attempt ${retries} of ${maxRetries}`;
                                setTimeout(checkForResult, 3000);
                            } else {
                                resultText.textContent = 'Result not found after waiting. Please try again later.';
                                progressContainer.classList.add('hidden');
                            }
                        }
                    } catch (err) {
                        console.error('Error fetching result JSON:', err);
                        resultText.textContent = 'Error fetching result JSON. See console.';
                        progressContainer.classList.add('hidden');
                    }
                };

                checkForResult();

            } catch (error) {
                console.error('Upload error:', error);
                resultText.textContent = `Upload error: ${error.message || 'Failed'}`;
                progressContainer.classList.add('hidden');
            } finally {
                uploadBtn.disabled = false;
            }
        });
    }
});

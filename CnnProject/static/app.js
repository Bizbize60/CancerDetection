document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    
    // Mode Selection
    const modeRadios = document.querySelectorAll('input[name="modelMode"]');
    
    // Sections
    const imageUploadSection = document.getElementById('imageUploadSection');
    const csvUploadSection = document.getElementById('csvUploadSection');
    
    // Image Upload Elements
    const imageDropZone = document.getElementById('imageDropZone');
    const imageInput = document.getElementById('imageInput');
    const imageUploadContent = document.getElementById('imageUploadContent');
    const imagePreviewContainer = document.getElementById('imagePreviewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const imageFileName = document.getElementById('imageFileName');
    const imageFileSize = document.getElementById('imageFileSize');
    const removeImageBtn = document.getElementById('removeImageBtn');
    
    // CSV Upload Elements
    const csvDropZone = document.getElementById('csvDropZone');
    const csvInput = document.getElementById('csvInput');
    const csvUploadContent = document.getElementById('csvUploadContent');
    const csvFileInfoContainer = document.getElementById('csvFileInfoContainer');
    const csvFileName = document.getElementById('csvFileName');
    const csvFileSize = document.getElementById('csvFileSize');
    const removeCsvBtn = document.getElementById('removeCsvBtn');
    
    // CSV Preview Elements
    const csvPreviewContainer = document.getElementById('csvPreviewContainer');
    const csvPreviewHead = document.getElementById('csvPreviewHead');
    const csvPreviewBody = document.getElementById('csvPreviewBody');
    const csvRowCountInfo = document.getElementById('csvRowCountInfo');
    const csvErrorMsg = document.getElementById('csvErrorMsg');
    
    // Form & Buttons
    const predictionForm = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictBtn');
    const btnText = document.getElementById('btnText');
    const btnIcon = document.getElementById('btnIcon');
    const btnSpinner = document.getElementById('btnSpinner');
    const formErrorMsg = document.getElementById('formErrorMsg');
    
    // Result Elements
    const resultEmptyState = document.getElementById('resultEmptyState');
    const resultContent = document.getElementById('resultContent');
    const batchResultContent = document.getElementById('batchResultContent');
    
    // Single Result Elements
    const resPredictedClass = document.getElementById('resPredictedClass');
    const resRiskBadge = document.getElementById('resRiskBadge');
    const resProbabilityText = document.getElementById('resProbabilityText');
    const resProgressBar = document.getElementById('resProgressBar');
    const resThreshold = document.getElementById('resThreshold');
    const resInterpretation = document.getElementById('resInterpretation');
    
    // Batch Result Elements
    const batchCountSummary = document.getElementById('batchCountSummary');
    const batchResultBody = document.getElementById('batchResultBody');
    
    // Accordion
    const accordionBtn = document.getElementById('accordionBtn');
    const accordionContent = document.getElementById('accordionContent');
    const accordionIcon = document.getElementById('accordionIcon');
    const techActiveMode = document.getElementById('techActiveMode');

    // State Variables
    let currentMode = 'fused';
    let selectedImageFile = null;
    let selectedCsvFile = null;
    let csvRowCount = 0; // Number of data rows

    // --- Mode Change Logic ---
    function updateModeVisibility() {
        if (currentMode === 'image') {
            imageUploadSection.style.display = 'block';
            csvUploadSection.style.display = 'none';
            techActiveMode.textContent = 'Image Only';
        } else if (currentMode === 'tabular') {
            imageUploadSection.style.display = 'none';
            csvUploadSection.style.display = 'block';
            techActiveMode.textContent = 'Tabular CSV Only';
        } else {
            imageUploadSection.style.display = 'block';
            csvUploadSection.style.display = 'block';
            techActiveMode.textContent = 'Image + Tabular Fusion';
        }
        formErrorMsg.classList.add('hidden');
    }

    modeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            currentMode = e.target.value;
            updateModeVisibility();
        });
    });

    // --- Image Upload Logic ---
    
    // Trigger file input on click
    imageDropZone.addEventListener('click', (e) => {
        if(e.target !== removeImageBtn && !removeImageBtn.contains(e.target)) {
            imageInput.click();
        }
    });

    // Drag and Drop
    imageDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        imageDropZone.classList.add('drag-active');
    });
    imageDropZone.addEventListener('dragleave', () => {
        imageDropZone.classList.remove('drag-active');
    });
    imageDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        imageDropZone.classList.remove('drag-active');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleImageFile(e.dataTransfer.files[0]);
        }
    });

    // File Input Change
    imageInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleImageFile(e.target.files[0]);
        }
    });

    function handleImageFile(file) {
        if (!file.type.match('image/jpeg') && !file.type.match('image/png')) {
            showError('Please upload only PNG or JPEG images.');
            return;
        }
        
        selectedImageFile = file;
        formErrorMsg.classList.add('hidden');

        // Update UI info
        imageFileName.textContent = file.name;
        imageFileSize.textContent = formatBytes(file.size);

        // Preview Image
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imageUploadContent.classList.add('hidden');
            imagePreviewContainer.classList.remove('hidden');
            imagePreviewContainer.classList.add('flex');
        };
        reader.readAsDataURL(file);
    }

    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedImageFile = null;
        imageInput.value = '';
        imagePreviewContainer.classList.add('hidden');
        imagePreviewContainer.classList.remove('flex');
        imageUploadContent.classList.remove('hidden');
    });


    // --- CSV Upload Logic ---

    // Trigger file input on click
    csvDropZone.addEventListener('click', (e) => {
        if(e.target !== removeCsvBtn && !removeCsvBtn.contains(e.target)) {
            csvInput.click();
        }
    });

    // Drag and Drop
    csvDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        csvDropZone.classList.add('drag-active');
    });
    csvDropZone.addEventListener('dragleave', () => {
        csvDropZone.classList.remove('drag-active');
    });
    csvDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        csvDropZone.classList.remove('drag-active');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleCsvFile(e.dataTransfer.files[0]);
        }
    });

    // File Input Change
    csvInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleCsvFile(e.target.files[0]);
        }
    });

    function handleCsvFile(file) {
        // Some systems don't have perfect mime types for CSV
        if (file.name.split('.').pop().toLowerCase() !== 'csv' && file.type !== 'text/csv') {
            showError('Please upload a valid CSV file.');
            return;
        }

        selectedCsvFile = file;
        formErrorMsg.classList.add('hidden');

        // Update UI info
        csvFileName.textContent = file.name;
        csvFileSize.textContent = formatBytes(file.size);

        csvUploadContent.classList.add('hidden');
        csvFileInfoContainer.classList.remove('hidden');
        csvFileInfoContainer.classList.add('flex');

        // Parse CSV for preview
        readCsvPreview(file);
    }

    removeCsvBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedCsvFile = null;
        csvRowCount = 0;
        csvInput.value = '';
        csvFileInfoContainer.classList.add('hidden');
        csvFileInfoContainer.classList.remove('flex');
        csvUploadContent.classList.remove('hidden');
        csvPreviewContainer.classList.add('hidden');
    });

    function readCsvPreview(file) {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            try {
                const text = e.target.result;
                const lines = text.split(/\r\n|\n/).filter(line => line.trim().length > 0);
                
                if (lines.length === 0) {
                    throw new Error("Empty CSV file");
                }

                // Simple comma split
                const headers = lines[0].split(',');
                csvRowCount = lines.length - 1;

                // Render Header
                csvPreviewHead.innerHTML = `<tr>${headers.map(h => `<th class="px-3 py-2 font-medium">${h}</th>`).join('')}</tr>`;

                // Render first 5 rows
                const maxRows = Math.min(5, lines.length - 1);
                let bodyHtml = '';
                for (let i = 1; i <= maxRows; i++) {
                    const cells = lines[i].split(',');
                    bodyHtml += `<tr class="border-b border-slate-100 hover:bg-slate-50">
                        ${cells.map(c => `<td class="px-3 py-2">${c}</td>`).join('')}
                    </tr>`;
                }
                csvPreviewBody.innerHTML = bodyHtml;

                csvPreviewContainer.classList.remove('hidden');
                csvErrorMsg.classList.add('hidden');

                if (csvRowCount > 1) {
                    csvRowCountInfo.classList.remove('hidden');
                } else {
                    csvRowCountInfo.classList.add('hidden');
                }

            } catch (err) {
                console.error(err);
                csvErrorMsg.classList.remove('hidden');
                csvRowCountInfo.classList.add('hidden');
                csvPreviewHead.innerHTML = '';
                csvPreviewBody.innerHTML = '';
            }
        };

        reader.onerror = () => {
            csvErrorMsg.classList.remove('hidden');
        };

        reader.readAsText(file);
    }


    // --- Accordion Logic ---
    accordionBtn.addEventListener('click', () => {
        accordionContent.classList.toggle('hidden');
        if (accordionContent.classList.contains('hidden')) {
            accordionIcon.style.transform = 'rotate(0deg)';
        } else {
            accordionIcon.style.transform = 'rotate(180deg)';
        }
    });

    // --- Form Submission & Prediction Logic ---
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Validate
        if (currentMode === 'image' || currentMode === 'fused') {
            if (!selectedImageFile) {
                showError('Mammography image is required for the selected mode.');
                return;
            }
        }
        if (currentMode === 'tabular' || currentMode === 'fused') {
            if (!selectedCsvFile) {
                showError('CSV metadata file is required for the selected mode.');
                return;
            }
        }

        formErrorMsg.classList.add('hidden');
        setLoadingState(true);

        // Prepare FormData
        const formData = new FormData();
        formData.append('mode', currentMode);
        if (selectedImageFile) formData.append('image', selectedImageFile);
        if (selectedCsvFile) formData.append('csv_file', selectedCsvFile);

        try {
            // Actual API Call (will likely fail if no backend)
            // const response = await fetch('/api/predict', { method: 'POST', body: formData });
            // if (!response.ok) throw new Error('API Error');
            // const data = await response.json();
            
            // Simulating API Call
            const data = await mockApiCall(currentMode, csvRowCount);
            displayResult(data);

        } catch (error) {
            console.error(error);
            showError('Prediction failed. Falling back to mock data...');
            // Fallback to mock on error
            setTimeout(async () => {
                const mockData = await mockApiCall(currentMode, csvRowCount);
                displayResult(mockData);
                formErrorMsg.classList.add('hidden'); // Clear error msg after fallback success
            }, 1000);
        } finally {
            setLoadingState(false);
        }
    });

    function displayResult(data) {
        resultEmptyState.classList.add('hidden');
        
        if (data.predictions && Array.isArray(data.predictions)) {
            // Batch Result
            resultContent.classList.add('hidden');
            batchResultContent.classList.remove('hidden');
            batchResultContent.classList.add('flex');
            
            batchCountSummary.textContent = `${data.count} rows processed`;
            
            let tbodyHtml = '';
            data.predictions.forEach(pred => {
                const colorClass = getRiskColorClass(pred.risk_level);
                tbodyHtml += `
                    <tr>
                        <td class="px-3 py-3 font-medium text-slate-800">#${pred.row_id}</td>
                        <td class="px-3 py-3">${(pred.probability * 100).toFixed(2)}%</td>
                        <td class="px-3 py-3">${pred.predicted_class}</td>
                        <td class="px-3 py-3 text-center">
                            <span class="px-2 py-1 rounded-full text-xs font-semibold ${colorClass}">${pred.risk_level}</span>
                        </td>
                    </tr>
                `;
            });
            batchResultBody.innerHTML = tbodyHtml;

        } else {
            // Single Result
            batchResultContent.classList.add('hidden');
            batchResultContent.classList.remove('flex');
            resultContent.classList.remove('hidden');
            resultContent.classList.add('flex');

            resPredictedClass.textContent = data.predicted_class;
            resProbabilityText.textContent = `${(data.probability * 100).toFixed(2)}%`;
            resThreshold.textContent = data.threshold;
            resInterpretation.textContent = data.interpretation;

            // Update Progress Bar
            setTimeout(() => {
                resProgressBar.style.width = `${data.probability * 100}%`;
                resProgressBar.className = `h-2.5 rounded-full transition-all duration-1000 ease-out ${getProgressBarColor(data.risk_level)}`;
            }, 50);

            // Update Badge
            resRiskBadge.textContent = data.risk_level;
            resRiskBadge.className = `px-3 py-1 rounded-full text-sm font-semibold border shadow-sm ${getRiskBadgeColorClass(data.risk_level)}`;
        }
    }

    // --- Mock API Logic ---
    function mockApiCall(mode, rowCount) {
        return new Promise((resolve) => {
            setTimeout(() => {
                if (mode === 'tabular' || mode === 'fused') {
                    if (rowCount > 1) {
                        // Generate mock batch
                        const preds = [];
                        for (let i = 1; i <= rowCount; i++) {
                            const p = Math.random();
                            let level = 'Low';
                            let cls = 'Benign / Low Risk';
                            if (p > 0.75) { level = 'High'; cls = 'Malignant Risk'; }
                            else if (p > 0.4) { level = 'Moderate'; cls = 'Moderate Risk'; }
                            
                            preds.push({ row_id: i, probability: p, predicted_class: cls, risk_level: level });
                        }
                        resolve({
                            mode: mode,
                            count: rowCount,
                            predictions: preds,
                            threshold: 0.5,
                            disclaimer: "This AI result is for research and decision-support purposes only."
                        });
                        return;
                    }
                }
                
                // Single Mock
                const p = Math.random() * 0.6 + 0.3; // Random between 0.3 and 0.9
                let level = 'Moderate';
                let cls = 'Moderate Risk';
                let interp = "The model predicts intermediate malignancy risk. Further investigation is recommended.";

                if (p > 0.75) {
                    level = 'High';
                    cls = 'Malignant Risk';
                    interp = "The model predicts elevated malignancy risk. This result is not a medical diagnosis and should be reviewed by a medical professional.";
                } else if (p < 0.4) {
                    level = 'Low';
                    cls = 'Benign / Low Risk';
                    interp = "The model predicts low malignancy risk. Routine screening is advised.";
                }

                resolve({
                    mode: mode,
                    probability: p,
                    predicted_class: cls,
                    risk_level: level,
                    threshold: 0.5,
                    interpretation: interp,
                    disclaimer: "This AI result is for research and decision-support purposes only."
                });

            }, 1500); // simulate delay
        });
    }

    // --- Utility Functions ---
    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    function showError(msg) {
        formErrorMsg.textContent = msg;
        formErrorMsg.classList.remove('hidden');
    }

    function setLoadingState(isLoading) {
        predictBtn.disabled = isLoading;
        if (isLoading) {
            btnText.textContent = 'Analyzing...';
            btnIcon.classList.add('hidden');
            btnSpinner.classList.remove('hidden');
        } else {
            btnText.textContent = 'Run AI Prediction';
            btnIcon.classList.remove('hidden');
            btnSpinner.classList.add('hidden');
        }
    }

    function getRiskColorClass(level) {
        if (level === 'High') return 'bg-red-100 text-red-700 border-red-200';
        if (level === 'Moderate') return 'bg-amber-100 text-amber-700 border-amber-200';
        return 'bg-green-100 text-green-700 border-green-200';
    }

    function getRiskBadgeColorClass(level) {
        if (level === 'High') return 'bg-red-50 text-red-600 border-red-200';
        if (level === 'Moderate') return 'bg-amber-50 text-amber-600 border-amber-200';
        return 'bg-green-50 text-green-600 border-green-200';
    }

    function getProgressBarColor(level) {
        if (level === 'High') return 'bg-red-500';
        if (level === 'Moderate') return 'bg-amber-500';
        return 'bg-green-500';
    }

    // Initialize UI
    updateModeVisibility();

});

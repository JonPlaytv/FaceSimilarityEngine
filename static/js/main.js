// Face Similarity Search - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initializeUploadHandlers();
    initializeDatasetButtons();
    checkDatasetStatus();
    checkDatasetStatusPeriodically();
});

function initializeDatasetButtons() {
    const crawlBtn = document.getElementById('crawlBtn');
    const initializeBtn = document.getElementById('initializeBtn');
    const clearUrlBtn = document.getElementById('clearUrlBtn');
    
    // Prevent multiple initializations
    if (crawlBtn && !crawlBtn.dataset.initialized) {
        crawlBtn.addEventListener('click', function() {
            const customUrlInput = document.getElementById('customUrlInput');
            const customUrl = customUrlInput ? customUrlInput.value.trim() : '';
            startCrawling(customUrl);
        });
        crawlBtn.dataset.initialized = 'true';
    }
    
    if (initializeBtn && !initializeBtn.dataset.initialized) {
        initializeBtn.addEventListener('click', initializeDataset);
        initializeBtn.dataset.initialized = 'true';
    }
    
    if (clearUrlBtn && !clearUrlBtn.dataset.initialized) {
        clearUrlBtn.addEventListener('click', function() {
            const customUrlInput = document.getElementById('customUrlInput');
            if (customUrlInput) {
                customUrlInput.value = '';
            }
        });
        clearUrlBtn.dataset.initialized = 'true';
    }
}

// Definiere die Handler-Funktion **außerhalb** der Initialisierung
function onFileInputChange(e) {
    handleFileSelection(e.target.files[0]);
}

function initializeUploadHandlers() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const chooseFileBtn = document.getElementById('chooseFileBtn');

    if (!uploadArea || !fileInput) return;

    // Prüfe, ob schon initialisiert
    if (fileInput.dataset.initialized === 'true') return;
    fileInput.dataset.initialized = 'true';

    // **WICHTIG:** Keine removeEventListener mehr nötig, weil dieser Handler nur einmal hinzugefügt wird
    fileInput.addEventListener('change', onFileInputChange);

    if (chooseFileBtn) {
        chooseFileBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation(); // Prevent bubbling to upload area
            fileInput.click();
        });
    }

    // Ensure submit button works properly
    if (submitBtn) {
        submitBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation(); // Prevent bubbling to upload area
            if (fileInput.files[0]) {
                uploadForm.submit();
            } else {
                showAlert('Please select a file first.', 'warning');
            }
        });
    }

    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelection(files[0]);
        }
    });

    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            if (!fileInput.files[0]) {
                e.preventDefault();
                showAlert('Please select a file first.', 'warning');
                return;
            }
            showProgress();
        });
    }

    uploadArea.addEventListener('click', function(e) {
        // Don't trigger file input if clicking on buttons or form elements
        if (e.target.tagName === 'BUTTON' || 
            e.target.tagName === 'INPUT' || 
            e.target.closest('button') || 
            e.target.closest('.btn')) {
            return;
        }
        fileInput.click();
    });
}

function handleFileSelection(file) {
    if (!file) return;

    // Validate file type
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp', 'image/avif', 'video/webm'];
    if (!allowedTypes.includes(file.type)) {
        showAlert('Please select a valid image file (PNG, JPG, JPEG, GIF, BMP, WebP, AVIF, WebM).', 'error');
        clearFile();
        return;
    }

    // Validate file size (16MB)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showAlert('File is too large. Please select an image smaller than 16MB.', 'error');
        clearFile();
        return;
    }

    // Show file info
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const submitBtn = document.getElementById('submitBtn');

    if (fileInfo && fileName) {
        fileName.textContent = `${file.name} (${formatFileSize(file.size)})`;
        fileInfo.style.display = 'block';
    }

    // Hide upload area text when file is selected
    const uploadAreaTitle = document.querySelector('.upload-area h5');
    const uploadAreaText = document.querySelector('.upload-area p');
    if (uploadAreaTitle) uploadAreaTitle.style.display = 'none';
    if (uploadAreaText) uploadAreaText.style.display = 'none';

    // Enable submit button when file is selected
    if (submitBtn) {
        submitBtn.disabled = false;
    }

    // Preview image (optional)
    previewImage(file);
}

function clearFile() {
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const submitBtn = document.getElementById('submitBtn');
    const imagePreview = document.getElementById('imagePreview');

    if (fileInput) fileInput.value = '';
    if (fileInfo) fileInfo.style.display = 'none';
    if (submitBtn) submitBtn.disabled = true;
    if (imagePreview) imagePreview.style.display = 'none';
    
    // Show upload area text
    const uploadAreaTitle = document.querySelector('.upload-area h5');
    const uploadAreaText = document.querySelector('.upload-area p');
    if (uploadAreaTitle) uploadAreaTitle.style.display = 'block';
    if (uploadAreaText) uploadAreaText.style.display = 'block';
}

function previewImage(file) {
    // Create image preview if container exists
    let previewContainer = document.getElementById('imagePreview');
    
    if (!previewContainer) {
        previewContainer = document.createElement('div');
        previewContainer.id = 'imagePreview';
        previewContainer.className = 'mt-3 text-center';
        previewContainer.style.display = 'none';
        
        const fileInfo = document.getElementById('fileInfo');
        if (fileInfo) {
            fileInfo.parentNode.insertBefore(previewContainer, fileInfo.nextSibling);
        }
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        previewContainer.innerHTML = `
            <div class="mb-2">
                <strong class="text-muted">Preview:</strong>
            </div>
            <img src="${e.target.result}" alt="Preview" class="img-fluid" style="max-height: 200px; border-radius: 8px;">
        `;
        previewContainer.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function showProgress() {
    const progressContainer = document.querySelector('.progress-container');
    const submitBtn = document.getElementById('submitBtn');

    if (progressContainer) {
        progressContainer.style.display = 'block';
        
        // Animate progress bar
        const progressBar = progressContainer.querySelector('.progress-bar');
        if (progressBar) {
            let width = 0;
            const interval = setInterval(() => {
                width += Math.random() * 15;
                if (width > 90) width = 90; // Don't complete until actual completion
                progressBar.style.width = width + '%';
            }, 200);
            
            // Store interval for cleanup
            progressContainer.dataset.interval = interval;
        }
    }

    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
    }
}

async function checkDatasetStatus() {
    try {
        const response = await fetch('/dataset_status');
        const data = await response.json();
        
        updateDatasetStatusUI(data);
        updateMainDatasetUI(data);
        
        return data.initialized && data.total_faces > 0;
    } catch (error) {
        console.error('Error checking dataset status:', error);
        updateDatasetStatusUI({initialized: false, size: 0});
        return false;
    }
}

function updateMainDatasetUI(data) {
    const datasetInfo = document.getElementById('datasetInfo');
    const initializeBtn = document.getElementById('initializeBtn');
    const crawlBtn = document.getElementById('crawlBtn');
    const advancedStats = document.getElementById('advancedStats');
    
    if (data.initialized && data.total_faces > 0) {
        if (datasetInfo) {
            datasetInfo.innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle me-2"></i>
                    Dataset ready with <strong>${data.total_faces}</strong> faces from <strong>${data.total_images}</strong> images
                    <br><small>Database: ${data.database_type || 'PostgreSQL + DuckDB'}</small>
                </div>
            `;
        }
        if (initializeBtn) initializeBtn.disabled = false;
        if (crawlBtn) crawlBtn.disabled = false;
        
        // Show advanced statistics
        const totalImages = document.getElementById('totalImages');
        const totalFaces = document.getElementById('totalFaces');
        const avgConfidence = document.getElementById('avgConfidence');
        const crawledCount = document.getElementById('crawledCount');
        
        if (totalImages) totalImages.textContent = data.total_images || 0;
        if (totalFaces) totalFaces.textContent = data.total_faces || 0;
        if (avgConfidence) avgConfidence.textContent = Math.round((data.avg_confidence || 0) * 100) + '%';
        if (crawledCount) crawledCount.textContent = data.crawl_sessions || 0;
        if (advancedStats) advancedStats.style.display = 'block';
        
    } else {
        if (datasetInfo) {
            datasetInfo.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    No face dataset found. Click "Crawl Images" to automatically collect face images from public websites, or "Initialize Dataset" to set up the system.
                    <br><small>The system needs images to compare against your uploads.</small>
                </div>
            `;
        }
        if (initializeBtn) initializeBtn.disabled = false;
        if (crawlBtn) crawlBtn.disabled = false;
    }
}

function updateDatasetStatusUI(data) {
    const statusBadge = document.getElementById('datasetStatusBadge');
    const statusText = document.getElementById('datasetStatusText');

    if (!statusBadge || !statusText) return;

    if (data.initialized && data.total_faces > 0) {
        statusBadge.className = 'badge bg-success';
        statusText.textContent = `${data.total_faces} faces ready`;
    } else {
        statusBadge.className = 'badge bg-warning';
        statusText.textContent = 'Dataset not initialized';
    }
}

function checkDatasetStatusPeriodically() {
    // Check dataset status every 30 seconds
    setInterval(checkDatasetStatus, 30000);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showAlert(message, type = 'info') {
    // Create alert element
    const alert = document.createElement('div');
    alert.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
    alert.innerHTML = `
        <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : type === 'success' ? 'check-circle' : 'info-circle'} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    // Insert at top of main content
    const main = document.querySelector('main.container');
    if (main) {
        main.insertBefore(alert, main.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 5000);
    }
}

// Utility function to handle image loading errors
function handleImageError(img) {
    img.style.display = 'none';
    const placeholder = document.createElement('div');
    placeholder.className = 'bg-secondary text-white d-flex align-items-center justify-content-center';
    placeholder.style.width = img.style.width || '100px';
    placeholder.style.height = img.style.height || '100px';
    placeholder.style.borderRadius = '8px';
    placeholder.innerHTML = '<i class="fas fa-image"></i>';
    
    img.parentNode.insertBefore(placeholder, img);
}

function startCrawling(customUrl = '') {
    const crawlBtn = document.getElementById('crawlBtn');
    const originalText = crawlBtn.innerHTML;
    
    crawlBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Crawling Images...';
    crawlBtn.disabled = true;
    
    fetch('/crawl_images', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            max_images: 50,
            custom_url: customUrl
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const alert = document.createElement('div');
            alert.className = 'alert alert-success alert-dismissible fade show mt-3';
            alert.innerHTML = `
                <i class="fas fa-check-circle me-2"></i>
                Successfully started crawling! Found ${data.images_crawled || 0} images with ${data.faces_detected || 0} faces.
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            document.getElementById('datasetCard').appendChild(alert);
            
            // Refresh status after crawling
            setTimeout(() => {
                checkDatasetStatus();
            }, 2000);
        } else {
            const alert = document.createElement('div');
            alert.className = 'alert alert-warning alert-dismissible fade show mt-3';
            alert.innerHTML = `
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${data.message || 'Crawling completed with some issues. Check the logs for details.'}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            document.getElementById('datasetCard').appendChild(alert);
        }
    })
    .catch(error => {
        console.error('Error crawling images:', error);
        const alert = document.createElement('div');
        alert.className = 'alert alert-danger alert-dismissible fade show mt-3';
        alert.innerHTML = `
            <i class="fas fa-exclamation-circle me-2"></i>
            Error starting image crawling. Please try again.
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.getElementById('datasetCard').appendChild(alert);
    })
    .finally(() => {
        crawlBtn.innerHTML = originalText;
        crawlBtn.disabled = false;
    });
}

function initializeDataset() {
    const initializeBtn = document.getElementById('initializeBtn');
    const originalText = initializeBtn.innerHTML;
    
    initializeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing Images...';
    initializeBtn.disabled = true;
    
    fetch('/initialize_dataset', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const alert = document.createElement('div');
            alert.className = 'alert alert-success alert-dismissible fade show mt-3';
            alert.innerHTML = `
                <i class="fas fa-check-circle me-2"></i>
                ${data.message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            document.getElementById('datasetCard').appendChild(alert);
            
            // Refresh status after initialization
            setTimeout(() => {
                checkDatasetStatus();
            }, 2000);
        } else {
            const alert = document.createElement('div');
            alert.className = 'alert alert-warning alert-dismissible fade show mt-3';
            alert.innerHTML = `
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${data.message || 'Dataset initialization had some issues.'}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            document.getElementById('datasetCard').appendChild(alert);
        }
    })
    .catch(error => {
        console.error('Error initializing dataset:', error);
        const alert = document.createElement('div');
        alert.className = 'alert alert-danger alert-dismissible fade show mt-3';
        alert.innerHTML = `
            <i class="fas fa-exclamation-circle me-2"></i>
            Error initializing dataset. Please try again.
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.getElementById('datasetCard').appendChild(alert);
    })
    .finally(() => {
        initializeBtn.innerHTML = originalText;
        initializeBtn.disabled = false;
    });
}



function clearFile() {
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const submitBtn = document.getElementById('submitBtn');
    const imagePreview = document.getElementById('imagePreview');

    if (fileInput) fileInput.value = '';
    if (fileInfo) fileInfo.style.display = 'none';
    if (submitBtn) submitBtn.disabled = true;
    if (imagePreview) imagePreview.style.display = 'none';
    
    // Show upload area text
    const uploadAreaTitle = document.querySelector('.upload-area h5');
    const uploadAreaText = document.querySelector('.upload-area p');
    if (uploadAreaTitle) uploadAreaTitle.style.display = 'block';
    if (uploadAreaText) uploadAreaText.style.display = 'block';
}

// Add global error handler for images
document.addEventListener('error', function(e) {
    if (e.target.tagName === 'IMG') {
        handleImageError(e.target);
    }
}, true);

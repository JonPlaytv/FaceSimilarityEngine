// Face Similarity Search - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initializeUploadHandlers();
    checkDatasetStatusPeriodically();
});

function initializeUploadHandlers() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');

    if (!uploadArea || !fileInput) return;

    // File input change handler
    fileInput.addEventListener('change', function(e) {
        handleFileSelection(e.target.files[0]);
    });

    // Drag and drop handlers
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

    // Form submission handler
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

    // Click to upload
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
}

function handleFileSelection(file) {
    if (!file) return;

    // Validate file type
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        showAlert('Please select a valid image file (PNG, JPG, JPEG, GIF, BMP).', 'error');
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

    // Enable submit button if dataset is ready
    checkDatasetStatus().then(isReady => {
        if (submitBtn) {
            submitBtn.disabled = !isReady;
        }
    });

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
        
        return data.initialized && data.size > 0;
    } catch (error) {
        console.error('Error checking dataset status:', error);
        return false;
    }
}

function updateDatasetStatusUI(data) {
    const statusBadge = document.getElementById('datasetStatusBadge');
    const statusText = document.getElementById('datasetStatusText');

    if (!statusBadge || !statusText) return;

    if (data.initialized && data.size > 0) {
        statusBadge.className = 'badge bg-success';
        statusText.textContent = `${data.size} faces ready`;
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

// Add global error handler for images
document.addEventListener('error', function(e) {
    if (e.target.tagName === 'IMG') {
        handleImageError(e.target);
    }
}, true);

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const videoUpload = document.getElementById('video-upload');
    const videoPreview = document.getElementById('video-preview');
    const videoPlaceholder = document.getElementById('video-placeholder');
    const recordBtn = document.getElementById('record-btn');
    const processBtn = document.getElementById('process-btn');
    const transcriptionEl = document.getElementById('transcription');
    const aiResponseEl = document.getElementById('ai-response');
    const audioPlayer = document.getElementById('audio-player');
    const statusBar = document.getElementById('status-bar');
    
    let wavesurfer = null;
    let currentSessionId = null;
    let mediaRecorder = null;
    let recordedChunks = [];
    let stream = null;
    let isRecording = false;
    
    // Initialize WaveSurfer
    function initWaveSurfer() {
        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: '#F59E0B',
            progressColor: '#FBBF24',
            cursorColor: '#FFFFFF',
            barWidth: 2,
            barRadius: 3,
            cursorWidth: 1,
            height: 100,
            barGap: 2,
            responsive: true
        });
        
        wavesurfer.on('ready', function() {
            wavesurfer.play();
        });
    }
    
    initWaveSurfer();
    
    // Update status bar
    function updateStatus(message, isError = false) {
        statusBar.textContent = message;
        statusBar.style.color = isError ? 'var(--danger)' : 'var(--gray)';
    }
    
    // Show error message
    function showError(message) {
        const errorEl = document.createElement('div');
        errorEl.className = 'error-message';
        errorEl.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
        document.body.appendChild(errorEl);
        
        setTimeout(() => {
            errorEl.style.opacity = '0';
            setTimeout(() => errorEl.remove(), 300);
        }, 5000);
    }
    
    // Handle video upload
    videoUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            handleVideoFile(file);
        }
    });
    
    // Handle drag and drop
    videoPlaceholder.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    });
    
    videoPlaceholder.addEventListener('dragleave', function() {
        this.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    });
    
    videoPlaceholder.addEventListener('drop', function(e) {
        e.preventDefault();
        this.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('video/')) {
            handleVideoFile(file);
        } else {
            showError('Please upload a video file');
        }
    });
    
    // Click on placeholder to trigger file input
    videoPlaceholder.addEventListener('click', function() {
        videoUpload.click();
    });
    
    // Handle video file
    function handleVideoFile(file) {
        if (!file.type.startsWith('video/')) {
            showError('Please select a video file');
            return;
        }
        
        const videoURL = URL.createObjectURL(file);
        showVideoPreview(videoURL);
        uploadVideo(file);
    }
    
    // Show video preview
    function showVideoPreview(url) {
        videoPreview.src = url;
        videoPreview.style.display = 'block';
        videoPlaceholder.style.display = 'none';
        processBtn.disabled = false;
        updateStatus('Video ready for processing');
    }
    
    // Upload video to server
    async function uploadVideo(file) {
        updateStatus('Uploading video...');
        
        const formData = new FormData();
        formData.append('video', file);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            if (response.ok) {
                currentSessionId = data.session_id;
                updateStatus('Video uploaded successfully');
            } else {
                showError(data.error || 'Failed to upload video');
                resetVideoUI();
            }
        } catch (error) {
            showError('Network error: ' + error.message);
            resetVideoUI();
        }
    }
    
    // Process video
    processBtn.addEventListener('click', async function() {
        if (!currentSessionId) {
            showError('No video to process');
            return;
        }
        
        processBtn.disabled = true;
        processBtn.innerHTML = '<span class="loading"></span> Processing...';
        updateStatus('Processing video...');
        
        // Clear previous results
        transcriptionEl.textContent = 'Processing...';
        aiResponseEl.textContent = 'Processing...';
        audioPlayer.src = '';
        wavesurfer.load('');
        
        try {
            const response = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: currentSessionId
                })
            });
            
            const data = await response.json();
            if (response.ok) {
                displayResults(data);
                updateStatus('Processing complete');
            } else {
                showError(data.error || 'Failed to process video');
            }
        } catch (error) {
            showError('Network error: ' + error.message);
        } finally {
            processBtn.disabled = false;
            processBtn.innerHTML = '<i class="fas fa-cogs"></i> Process Video';
        }
    });
    
    // Display results
    function displayResults(data) {
        transcriptionEl.textContent = data.transcription || 'No transcription available';
        aiResponseEl.textContent = data.response || 'No response generated';
        
        if (data.audio_url) {
            audioPlayer.src = data.audio_url;
            wavesurfer.load(data.audio_url);
        }
    }
    
    // Reset video UI
    function resetVideoUI() {
        videoPreview.src = '';
        videoPreview.style.display = 'none';
        videoPlaceholder.style.display = 'flex';
        processBtn.disabled = true;
        currentSessionId = null;
    }
    
    // Video recording
    recordBtn.addEventListener('click', async function() {
        if (isRecording) {
            stopRecording();
            return;
        }
        
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: true, 
                audio: true 
            });
            
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'video/webm'
            });
            
            recordedChunks = [];
            
            mediaRecorder.ondataavailable = function(e) {
                if (e.data.size > 0) {
                    recordedChunks.push(e.data);
                }
            };
            
            mediaRecorder.onstop = function() {
                const blob = new Blob(recordedChunks, { type: 'video/webm' });
                const videoURL = URL.createObjectURL(blob);
                showVideoPreview(videoURL);
                uploadVideo(blob);
                recordedChunks = [];
            };
            
            // Start recording
            mediaRecorder.start(100); // Collect data every 100ms
            isRecording = true;
            recordBtn.innerHTML = '<span class="recording-indicator"></span> Stop Recording';
            recordBtn.classList.add('btn-danger');
            recordBtn.classList.remove('btn-gradient');
            videoPreview.srcObject = stream;
            videoPreview.style.display = 'block';
            videoPlaceholder.style.display = 'none';
            updateStatus('Recording... Press again to stop');
        } catch (error) {
            showError('Error accessing camera: ' + error.message);
            isRecording = false;
            resetRecordButton();
        }
    });
    
    // Stop recording
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        isRecording = false;
        resetRecordButton();
    }
    
    // Reset record button
    function resetRecordButton() {
        recordBtn.innerHTML = '<i class="fas fa-circle"></i> Record Video';
        recordBtn.classList.remove('btn-danger');
        recordBtn.classList.add('btn-gradient');
    }
    
    // Handle page before unload
    window.addEventListener('beforeunload', function() {
        if (isRecording) {
            stopRecording();
        }
    });
});
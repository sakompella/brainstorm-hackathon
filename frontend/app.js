/**
 * Neural Data Viewer - WebSocket Client (Middleware Mode)
 *
 * Connects to middleware feature server and renders bandpower heatmaps
 */

// Theme colormap (256 values, RGB)
const MAGMA_COLORMAP = generateMagmaColormap();

// State
let ws = null;
let gridSize = 32;
let isConnected = false;

// FPS tracking
let frameCount = 0;
let lastFpsUpdate = performance.now();
let currentFps = 0;

// Time tracking
let currentTime = 0.0;

// Baseline for dB conversion
let baselineMeanPower = null;
const DB_EPS = 1e-12;

// Time series data
let timeSeriesData = [];
let maxTimeSeriesPoints = 500;
let timeSeriesCanvas = null;
let timeSeriesCtx = null;

// Canvas and rendering
let canvas = null;
let ctx = null;
let canvasWidth = 32;
let canvasHeight = 32;

// Value range for colormap (auto-scaled from data)
let vMin = 0.0;
let vMax = 0.01;

/**
 * Resize canvas to fit container.
 */
function resizeCanvasToContainer() {
    if (!canvas) return;
    const container = canvas.parentElement;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const minSide = Math.min(rect.width, rect.height);
    const size = Math.max(120, Math.floor((minSide > 0 ? minSide : 360) * 0.85));
    const dpr = window.devicePixelRatio || 1;

    canvas.width = Math.floor(size * dpr);
    canvas.height = Math.floor(size * dpr);
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;

    canvasWidth = size;
    canvasHeight = size;

    if (ctx) {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);
    }
}

/**
 * Generate the magma colormap as an array of [r, g, b] values.
 */
function generateMagmaColormap() {
    const magmaData = [
        [0.001462, 0.000466, 0.013866],
        [0.013708, 0.011771, 0.068667],
        [0.039608, 0.031090, 0.133515],
        [0.074257, 0.052017, 0.193510],
        [0.113094, 0.065492, 0.243537],
        [0.154901, 0.071327, 0.284065],
        [0.198177, 0.072245, 0.316356],
        [0.241397, 0.072699, 0.340836],
        [0.284124, 0.073417, 0.358296],
        [0.326438, 0.074167, 0.369846],
        [0.368567, 0.074621, 0.376400],
        [0.410791, 0.074866, 0.378497],
        [0.453187, 0.074686, 0.376427],
        [0.495784, 0.074295, 0.370369],
        [0.538516, 0.073859, 0.360437],
        [0.581246, 0.073480, 0.346753],
        [0.623796, 0.073307, 0.329512],
        [0.666022, 0.073590, 0.308947],
        [0.707797, 0.074578, 0.285380],
        [0.748980, 0.076556, 0.259246],
        [0.789417, 0.079868, 0.230962],
        [0.828991, 0.084937, 0.200963],
        [0.867534, 0.092252, 0.169642],
        [0.904837, 0.102306, 0.137338],
        [0.940621, 0.115594, 0.104286],
        [0.974449, 0.133635, 0.070619],
        [0.995560, 0.165380, 0.039886],
        [0.998085, 0.211843, 0.021563],
        [0.987053, 0.266188, 0.024335],
        [0.968443, 0.321898, 0.042144],
        [0.948683, 0.375586, 0.064264],
        [0.932067, 0.426710, 0.088087],
        [0.921248, 0.475767, 0.111534],
        [0.917482, 0.523424, 0.133798],
        [0.920858, 0.570213, 0.154815],
        [0.931674, 0.616411, 0.175091],
        [0.949545, 0.662198, 0.195563],
        [0.973381, 0.707719, 0.217587],
        [0.993248, 0.753418, 0.243755],
        [0.998364, 0.800551, 0.282327],
        [0.987622, 0.849251, 0.337977],
        [0.969680, 0.897560, 0.410320],
        [0.963855, 0.941167, 0.490000],
        [0.980600, 0.973500, 0.560100],
        [0.987053, 0.991438, 0.749504]
    ];

    const colormap = [];
    for (let i = 0; i < 256; i++) {
        const t = i / 255 * (magmaData.length - 1);
        const idx = Math.floor(t);
        const frac = t - idx;

        if (idx >= magmaData.length - 1) {
            const c = magmaData[magmaData.length - 1];
            colormap.push([
                Math.round(c[0] * 255),
                Math.round(c[1] * 255),
                Math.round(c[2] * 255)
            ]);
        } else {
            const c1 = magmaData[idx];
            const c2 = magmaData[idx + 1];
            colormap.push([
                Math.round((c1[0] + frac * (c2[0] - c1[0])) * 255),
                Math.round((c1[1] + frac * (c2[1] - c1[1])) * 255),
                Math.round((c1[2] + frac * (c2[2] - c1[2])) * 255)
            ]);
        }
    }

    return colormap;
}

/**
 * Map a value to a colormap index.
 */
function valueToColorIndex(value) {
    const normalized = (value - vMin) / (vMax - vMin);
    const clamped = Math.max(0, Math.min(1, normalized));
    return Math.round(clamped * 255);
}

/**
 * Get RGB color string for a value.
 */
function valueToColor(value) {
    const idx = valueToColorIndex(value);
    const [r, g, b] = MAGMA_COLORMAP[idx];
    return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Initialize the canvas.
 */
function initCanvas() {
    canvas = document.getElementById('neural-canvas');
    ctx = canvas.getContext('2d');
    resizeCanvasToContainer();
    
    // Initialize time series canvas
    timeSeriesCanvas = document.getElementById('timeseries-canvas');
    if (timeSeriesCanvas) {
        timeSeriesCtx = timeSeriesCanvas.getContext('2d');
        resizeTimeSeriesCanvas();
    }
}

/**
 * Resize time series canvas to fit container.
 */
function resizeTimeSeriesCanvas() {
    if (!timeSeriesCanvas) return;
    const container = timeSeriesCanvas.parentElement;
    if (!container) return;
    
    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    
    timeSeriesCanvas.width = Math.floor(rect.width * dpr);
    timeSeriesCanvas.height = Math.floor(Math.min(200, rect.height) * dpr);
    timeSeriesCanvas.style.width = `${rect.width}px`;
    timeSeriesCanvas.style.height = `${Math.min(200, rect.height)}px`;
    
    if (timeSeriesCtx) {
        timeSeriesCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
}

/**
 * Render heatmap.
 */
function renderHeatmap(heatmap, centroid) {
    if (!heatmap || !ctx) return;

    const rows = heatmap.length;
    const cols = heatmap[0].length;

    // Auto-scale vMax based on data
    let maxVal = 0;
    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            if (heatmap[row][col] > maxVal) maxVal = heatmap[row][col];
        }
    }
    vMax = 0.9 * vMax + 0.1 * Math.max(maxVal, 0.001);

    const size = Math.min(canvasWidth, canvasHeight);
    const padding = Math.max(4, Math.floor(size * 0.04));
    const plotSize = Math.max(1, size - 2 * padding);
    const cellSize = plotSize / cols;

    // Clear canvas
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Draw heatmap
    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            const value = heatmap[row][col];
            const x = padding + col * cellSize;
            const y = padding + row * cellSize;

            const colorIdx = valueToColorIndex(value);
            const [r, g, b] = MAGMA_COLORMAP[colorIdx];
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(x, y, cellSize + 0.5, cellSize + 0.5);
        }
    }

    // Draw grid overlay
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    for (let i = 0; i <= cols; i++) {
        const offset = padding + i * cellSize;
        ctx.moveTo(offset, padding);
        ctx.lineTo(offset, padding + plotSize);
        ctx.moveTo(padding, offset);
        ctx.lineTo(padding + plotSize, offset);
    }
    ctx.stroke();

    // Draw centroid if provided
    if (centroid) {
        const [cy, cx] = centroid; // backend sends [y,x]
        ctx.fillStyle = 'cyan';
        ctx.beginPath();
        ctx.arc(
            padding + (cx + 0.5) * cellSize,
            padding + (cy + 0.5) * cellSize,
            Math.max(3, cellSize / 2),
            0,
            2 * Math.PI
        );
        ctx.fill();

        // --- Compute vector from center (16,16) to centroid ---
        const center = [16, 16];
        let vecX = cx - center[1];
        let vecY = cy - center[0];

        // Compute magnitude
        const mag = Math.sqrt(vecX ** 2 + vecY ** 2) || 1; // avoid div0
        const unitX = vecX / mag;
        const unitY = vecY / mag;

        // Draw arrow from center to centroid
        ctx.strokeStyle = 'lime';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padding + (center[1] + 0.5) * cellSize, padding + (center[0] + 0.5) * cellSize);
        ctx.lineTo(padding + (cx + 0.5) * cellSize, padding + (cy + 0.5) * cellSize);
        ctx.stroke();

        // --- Translate unit vector into neuro/medical directions ---
        let direction = '';
        if (Math.abs(unitY) < 0.3 && unitX > 0.3) direction = 'Right';
        else if (Math.abs(unitY) < 0.3 && unitX < -0.3) direction = 'Left';
        else if (Math.abs(unitX) < 0.3 && unitY > 0.3) direction = 'Posterior';
        else if (Math.abs(unitX) < 0.3 && unitY < -0.3) direction = 'Anterior';
        else if (unitX > 0 && unitY < 0) direction = 'Anterior-Right';
        else if (unitX < 0 && unitY < 0) direction = 'Anterior-Left';
        else if (unitX > 0 && unitY > 0) direction = 'Posterior-Right';
        else if (unitX < 0 && unitY > 0) direction = 'Posterior-Left';
        else direction = 'Center';

        // Display direction text
        ctx.fillStyle = 'white';
        ctx.font = '16px monospace';
        ctx.fillText(`Direction: ${direction}`, 10, 20);
    }

    // Update FPS counter
    frameCount++;
    const now = performance.now();
    if (now - lastFpsUpdate >= 1000) {
        currentFps = frameCount;
        frameCount = 0;
        lastFpsUpdate = now;
        document.getElementById('fps-counter').textContent = `${currentFps} FPS`;
    }
}


/**
 * Update connection status UI.
 */
function updateStatus(status, text) {
    const indicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const connectBtn = document.getElementById('connect-btn');

    indicator.className = 'status-indicator';

    switch (status) {
        case 'connected':
            indicator.classList.add('connected');
            statusText.textContent = 'Connected';
            connectBtn.textContent = 'Disconnect';
            connectBtn.classList.add('disconnect');
            connectBtn.disabled = false;
            isConnected = true;
            break;
        case 'connecting':
            indicator.classList.add('connecting');
            statusText.textContent = 'Connecting...';
            connectBtn.disabled = true;
            break;
        case 'disconnected':
        default:
            statusText.textContent = text || 'Disconnected';
            connectBtn.textContent = 'Connect';
            connectBtn.classList.remove('disconnect');
            connectBtn.disabled = false;
            isConnected = false;
            break;
    }
}

/**
 * Update info cards with feature data.
 */
function updateInfoCards(data) {
    // Update presence indicator
    const presence = data.presence || 0;
    const presenceCard = document.querySelector('.card-coverage .card-body');
    presenceCard.textContent = `Presence: ${presence.toFixed(3)}`;

    // Update confidence
    const confidence = data.confidence || 0;
    const statusCard = document.querySelector('.card-status .card-body');
    statusCard.textContent = confidence > 0.5 ? 'Online' : 'Offline';
}

/**
 * Plot time series data.
 */
function plotTimeSeries(meanPowerDb, timestamp) {
    if (!timeSeriesCtx || !timeSeriesCanvas) return;
    
    // Add new data point
    timeSeriesData.push({ t: timestamp, value: meanPowerDb });
    
    // Keep only recent data
    if (timeSeriesData.length > maxTimeSeriesPoints) {
        timeSeriesData.shift();
    }
    
    if (timeSeriesData.length < 2) return;
    
    const width = (timeSeriesCanvas.width / (window.devicePixelRatio || 1))-40;
    const height = timeSeriesCanvas.height / (window.devicePixelRatio || 1);
    const padding = { top: 20, right: 40, bottom: 30, left: 50 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    
    // Clear canvas
    timeSeriesCtx.fillStyle = '#0a0a0f';
    timeSeriesCtx.fillRect(0, 0, width, height);
    
    // Find data range
    const values = timeSeriesData.map(d => d.value);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const valueRange = maxValue - minValue || 1;
    
    // EKG-style: newest data at 80% of the plot width
    const latestTime = timeSeriesData[timeSeriesData.length - 1].t;
    const oldestTime = timeSeriesData[0].t;
    const actualTimeRange = latestTime - oldestTime || 1;
    
    // Scale time range so that the actual data spans 80% of the width
    const displayTimeRange = actualTimeRange / 0.85;
    
    // Draw axes
    timeSeriesCtx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    timeSeriesCtx.lineWidth = 1;
    timeSeriesCtx.beginPath();
    timeSeriesCtx.moveTo(padding.left, padding.top);
    timeSeriesCtx.lineTo(padding.left, height - padding.bottom);
    timeSeriesCtx.lineTo(width - padding.right, height - padding.bottom);
    timeSeriesCtx.stroke();
    
    // Draw grid lines
    timeSeriesCtx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    timeSeriesCtx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (plotHeight * i / 4);
        timeSeriesCtx.beginPath();
        timeSeriesCtx.moveTo(padding.left, y);
        timeSeriesCtx.lineTo(width - padding.right, y);
        timeSeriesCtx.stroke();
    }
    
    // Draw labels
    timeSeriesCtx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    timeSeriesCtx.font = '10px JetBrains Mono, monospace';
    timeSeriesCtx.textAlign = 'right';
    timeSeriesCtx.textBaseline = 'middle';
    
    for (let i = 0; i <= 4; i++) {
        const value = maxValue - (valueRange * i / 4);
        const y = padding.top + (plotHeight * i / 4);
        timeSeriesCtx.fillText(value.toFixed(2), padding.left - 5, y);
    }
    
    // Draw time series line (EKG-style, newest at 80% position)
    timeSeriesCtx.strokeStyle = '#ff6b6b';
    timeSeriesCtx.lineWidth = 2;
    timeSeriesCtx.beginPath();
    
    for (let i = 0; i < timeSeriesData.length; i++) {
        const d = timeSeriesData[i];
        // Position relative to latest time, with latest at 80% of plot width
        const timeFromLatest = latestTime - d.t;
        const normalizedPosition = 0.8 - (timeFromLatest / displayTimeRange);
        const x = padding.left + 60 + normalizedPosition * plotWidth;
        const y = height - padding.bottom - ((d.value - minValue) / valueRange) * plotHeight;
        
        if (i === 0) {
            timeSeriesCtx.moveTo(x, y);
        } else {
            timeSeriesCtx.lineTo(x, y);
        }
    }
    
    timeSeriesCtx.stroke();
    
    // Draw axis labels
    timeSeriesCtx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    timeSeriesCtx.font = '12px JetBrains Mono, monospace';
    timeSeriesCtx.textAlign = 'center';
    timeSeriesCtx.fillText('Time (s)', width / 2, height - 5);
    
    timeSeriesCtx.save();
    timeSeriesCtx.translate(10, height / 2);
    timeSeriesCtx.rotate(-Math.PI / 2);
    timeSeriesCtx.fillText('Mean Power (dB)', 0, 0);
    timeSeriesCtx.restore();
}

/**
 * Connect to WebSocket server.
 */
function connect() {
    const url = document.getElementById('server-url').value;

    if (isConnected && ws) {
        ws.close();
        return;
    }

    updateStatus('connecting');

    try {
        ws = new WebSocket(url);

        ws.onopen = () => {
            console.log('WebSocket connected to middleware');
            updateStatus('connected');
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.type === 'features') {
                    // Update time display
                    currentTime = data.t || 0;
                    document.getElementById('time-display').textContent = `t = ${currentTime.toFixed(2)}s`;

                    // Update channel count (grid size from heatmap)
                    const heatmap = data.heatmap;
                    const centroid = data.centroid;
                    if (heatmap && heatmap.length > 0) {
                        const dataSize = heatmap.length;
                        const nChannels = 96 * 96;
                        document.getElementById('channel-count').textContent = `${nChannels} channels (${dataSize}x${dataSize} data)`;

                        // Render heatmap
                        renderHeatmap(heatmap, centroid);

                        // Update info cards
                        updateInfoCards(data);
                        
                        // Calculate mean bandpower for time series
                        let sum = 0;
                        let count = 0;
                        for (let row = 0; row < heatmap.length; row++) {
                            for (let col = 0; col < heatmap[row].length; col++) {
                                sum += heatmap[row][col];
                                count++;
                            }
                        }
                        const meanPower = count > 0 ? sum / count : 0;
                        if (baselineMeanPower === null) {
                            baselineMeanPower = Math.max(meanPower, DB_EPS);
                        }
                        const dbPower = 10 * Math.log10(Math.max(meanPower, DB_EPS) / baselineMeanPower);
                        
                        // Plot time series
                        plotTimeSeries(dbPower, currentTime);
                    }
                }
            } catch (err) {
                console.error('Error parsing message:', err);
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateStatus('disconnected', 'Connection error');
        };

        ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            updateStatus('disconnected', event.wasClean ? 'Disconnected' : 'Connection lost');
            ws = null;
        };

    } catch (err) {
        console.error('Failed to create WebSocket:', err);
        updateStatus('disconnected', 'Failed to connect');
    }
}

/**
 * Initialize the application.
 */
function init() {
    initCanvas();
    window.addEventListener('resize', () => {
        resizeCanvasToContainer();
        resizeTimeSeriesCanvas();
    });

    // Connect button handler
    document.getElementById('connect-btn').addEventListener('click', connect);

    // Enter key in URL input
    document.getElementById('server-url').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            connect();
        }
    });
}

// Start when DOM is ready
document.addEventListener('DOMContentLoaded', init);

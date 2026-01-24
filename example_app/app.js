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

// Canvas and rendering
let canvas = null;
let ctx = null;
let canvasWidth = 32;
let canvasHeight = 32;

// Value range for colormap (log-normalized bandpower)
let vMin = -2.0;
let vMax = 2.0;

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
}

/**
 * Render heatmap from middleware features.
 */
function renderHeatmap(heatmap) {
    if (!heatmap || !ctx) return;

    const size = Math.min(canvasWidth, canvasHeight);
    const padding = Math.max(4, Math.floor(size * 0.04));
    const plotSize = Math.max(1, size - 2 * padding);
    const cellSize = plotSize / gridSize;

    // Clear canvas
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Draw heatmap
    for (let row = 0; row < gridSize; row++) {
        for (let col = 0; col < gridSize; col++) {
            const value = heatmap[row][col];
            const x = padding + col * cellSize;
            const y = padding + row * cellSize;

            ctx.fillStyle = valueToColor(value);
            ctx.fillRect(x, y, cellSize, cellSize);
        }
    }

    // Draw grid overlay
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= gridSize; i++) {
        const offset = padding + i * cellSize;
        ctx.moveTo(offset, padding);
        ctx.lineTo(offset, padding + plotSize);
        ctx.moveTo(padding, offset);
        ctx.lineTo(padding + plotSize, offset);
    }
    ctx.stroke();

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

    // Update band info
    const band = data.band || [70, 150];
    const monitorBody = document.querySelector('.monitor-body');
    monitorBody.textContent = `Bandpower: ${band[0]}-${band[1]} Hz | Samples: ${data.total_samples || 0}`;
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
                    if (heatmap && heatmap.length > 0) {
                        gridSize = heatmap.length;
                        const nChannels = gridSize * gridSize;
                        document.getElementById('channel-count').textContent = `${nChannels} channels`;

                        // Render heatmap
                        renderHeatmap(heatmap);

                        // Update info cards
                        updateInfoCards(data);
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
    window.addEventListener('resize', resizeCanvasToContainer);

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
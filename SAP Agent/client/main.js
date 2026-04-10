import { RetellWebClient } from 'retell-client-js-sdk';

const retellWebClient = new RetellWebClient();

const orb = document.getElementById('orb');
const orbWrapper = document.getElementById('sphereWrapper');
const visualizer = document.getElementById('visualizer');
const statusBadge = document.getElementById('status-text');
const ctaButton = document.getElementById('ctaButton');
const iconMic = document.getElementById('iconMic');
const iconPhoneOff = document.getElementById('iconPhoneOff');
const pingRing1 = document.getElementById('pingRing1');
const pingRing2 = document.getElementById('pingRing2');

const SERVER_URL = 'http://localhost:3000/api/create-web-call';

let isCallActive = false;

// --- Retell Event Listeners ---

retellWebClient.on('call_started', () => {
    console.log('Call started');
    updateUIState('active');
    statusBadge.textContent = 'Agent is listening...';
});

retellWebClient.on('call_ended', () => {
    console.log('Call ended');
    updateUIState('idle');
    statusBadge.textContent = 'Call ended';
    setTimeout(() => {
        if (!isCallActive) {
            statusBadge.textContent = 'Ready for your session. Click to start.';
        }
    }, 2000);
});

retellWebClient.on('error', (error) => {
    console.error('An error occurred:', error);
    updateUIState('idle');
    statusBadge.textContent = 'An error occurred. Please try again.';
    retellWebClient.stopCall();
});

retellWebClient.on('update', (update) => {
    // Optional: visualize audio updates
});

// --- UI Interactions ---

orb.addEventListener('click', toggleCall);
ctaButton.addEventListener('click', toggleCall);

// --- Core Functions ---

async function toggleCall() {
    // Ripple feedback on click
    const ripple = document.createElement('div');
    ripple.classList.add('ripple');
    orbWrapper.appendChild(ripple);
    requestAnimationFrame(() => ripple.classList.add('animate'));
    setTimeout(() => ripple.remove(), 1000);

    if (isCallActive) {
        stopCall();
    } else {
        startCall();
    }
}

async function startCall() {
    statusBadge.textContent = 'Connecting...';
    statusBadge.classList.add('active');
    orb.style.pointerEvents = 'none';
    ctaButton.style.pointerEvents = 'none';

    try {
        const response = await fetch(SERVER_URL);
        if (!response.ok) {
            throw new Error(`Server fetch failed: ${response.statusText}`);
        }
        const data = await response.json();

        if (!data.access_token) {
            throw new Error('No access token received from backend');
        }

        await retellWebClient.startCall({
            accessToken: data.access_token,
        });
        isCallActive = true;

    } catch (error) {
        console.error('Failed to start call:', error);
        statusBadge.textContent = 'Connection failed. Try again.';
        statusBadge.classList.remove('active');
        isCallActive = false;
    } finally {
        orb.style.pointerEvents = 'auto';
        ctaButton.style.pointerEvents = 'auto';
    }
}

function stopCall() {
    statusBadge.textContent = 'Disconnecting...';
    retellWebClient.stopCall();
    isCallActive = false;
}

function updateUIState(state) {
    if (state === 'active') {
        orb.classList.add('active');
        orbWrapper.classList.add('active');
        visualizer.classList.add('active');
        statusBadge.classList.add('active');
        // Show end-call icon
        ctaButton.classList.add('call-active');
        iconMic.style.display = 'none';
        iconPhoneOff.style.display = 'block';
    } else {
        orb.classList.remove('active');
        orbWrapper.classList.remove('active');
        visualizer.classList.remove('active');
        statusBadge.classList.remove('active');
        // Show mic icon
        ctaButton.classList.remove('call-active');
        iconMic.style.display = 'block';
        iconPhoneOff.style.display = 'none';
    }
}

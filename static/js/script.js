// ========================
// Config
// ========================
const API_BASE = "http://127.0.0.1:5000/api";

// ========================
// Utility Functions
// ========================
function showError(message) {
    const errorDiv = document.getElementById('error-message');
    if (errorDiv) {
        errorDiv.innerText = message;
        errorDiv.style.display = 'block';
    } else {
        alert(message);
    }
}

function hideError() {
    const errorDiv = document.getElementById('error-message');
    if (errorDiv) errorDiv.style.display = 'none';
}

// ========================
// Login Function
// ========================
async function login() {
    console.log("Login function called");

    // Hide any previous errors
    const errorDiv = document.getElementById('error-message');
    if (errorDiv) {
        errorDiv.style.display = 'none';
        errorDiv.textContent = '';
    }

    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    console.log("Login values:", {username, password});

    if (!username || !password) {
        showError('Please enter both username and password');
        return;
    }

    try {
        console.log("Sending login request to /api/login...");

        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username: username,
                password: password
            })
        });

        console.log("Response status:", response.status);

        // Check if response is JSON
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error('Server returned non-JSON response');
        }

        const data = await response.json();
        console.log("Response data:", data);

        if (response.ok) {
            console.log("Login successful, storing token...");
            localStorage.setItem('authToken', data.access_token);
            localStorage.setItem('currentUser', JSON.stringify(data.user));
            window.location.href = '/';
        } else {
            showError(data.error || 'Login failed. Please check your credentials.');
        }
    } catch (err) {
        console.error("Login error details:", err);
        showError('Server error. Please try again later. Error: ' + err.message);
    }
}

// ========================
// Register Function
// ========================
// In your register function in script.js
async function register() {
    hideError();

    const username = document.getElementById('username').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const firstName = document.getElementById('first_name').value;
    const lastName = document.getElementById('last_name').value;
    const phone = document.getElementById('phone').value;
    const emergencyContact = document.getElementById('emergency_contact').value;

    if (!username || !email || !password || !firstName || !lastName) {
        showError('Please fill all required fields');
        return;
    }

    try {
        console.log("Sending registration request..."); // Debug log

        const response = await fetch('/api/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username,
                email,
                password,
                first_name: firstName,
                last_name: lastName,
                phone,
                emergency_contact: emergencyContact
            })
        });

        console.log("Response status:", response.status); // Debug log

        const data = await response.json();
        console.log("Response data:", data); // Debug log

        if (response.ok) {
            alert('Registration successful. Please login.');
            window.location.href = '/login';
        } else {
            showError(data.error || 'Registration failed');
        }
    } catch (err) {
        console.error("Registration error:", err); // Debug log
        showError('Server error. Please try again later.');
    }
}

// ========================
// Location Update Function
// ========================
async function sendLocation(latitude, longitude) {
    const token = localStorage.getItem('authToken');
    if (!token) return;

    try {
        const response = await fetch(`${API_BASE}/location`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ latitude, longitude })
        });

        const data = await response.json();

        if (response.ok) {
            console.log("Location updated successfully");

            // Update UI with anomaly information
            if (document.getElementById('anomaly-score')) {
                document.getElementById('anomaly-score').textContent = data.anomaly_score.toFixed(2);
            }

            if (document.getElementById('current-status')) {
                if (data.is_anomaly || data.in_risk_zone) {
                    document.getElementById('current-status').textContent = 'Anomaly Detected!';
                    document.getElementById('current-status').style.color = '#e74c3c';
                } else {
                    document.getElementById('current-status').textContent = 'Normal';
                    document.getElementById('current-status').style.color = '#27ae60';
                }
            }

            // Refresh alerts if anomaly detected
            if (data.alert_triggered) {
                refreshAlerts();
            }
        } else {
            console.error("Location update failed:", data.error);
        }
    } catch (err) {
        console.error("Location update error:", err);
    }
}

// ========================
// Panic Button Function
// ========================
async function sendPanicAlert() {
    const token = localStorage.getItem('authToken');
    if (!token) {
        showError("Please login first");
        return;
    }

    try {
        // Try to get current location
        const position = await getCurrentLocation();

        const response = await fetch(`${API_BASE}/panic`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                latitude: position.coords.latitude,
                longitude: position.coords.longitude
            })
        });

        const data = await response.json();

        if (response.ok) {
            alert("Panic alert sent successfully!");
            refreshAlerts();
        } else {
            showError(data.error || "Failed to send panic alert");
        }
    } catch (err) {
        // If location access is denied, send without coordinates
        try {
            const response = await fetch(`${API_BASE}/panic`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                }
            });

            const data = await response.json();

            if (response.ok) {
                alert('Panic alert sent successfully! Help is on the way.');
                refreshAlerts();
            } else {
                showError(data.error || 'Error sending panic alert');
            }
        } catch (innerError) {
            showError('Error sending panic alert. Please try again.');
        }
    }
}

// ========================
// Alerts Polling Function
// ========================
async function refreshAlerts() {
    const token = localStorage.getItem('authToken');
    if (!token) return;

    try {
        const showResolved = document.getElementById('show-resolved') ?
                            document.getElementById('show-resolved').checked : false;

        const response = await fetch(`${API_BASE}/alerts?resolved=${showResolved}`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        const data = await response.json();

        if (response.ok) {
            displayAlerts(data.alerts);
            updateAlertStats(data.alerts);
        }
    } catch (err) {
        console.error("Error fetching alerts:", err);
    }
}

// ========================
// Display Alerts
// ========================
function displayAlerts(alerts) {
    const container = document.getElementById('alerts-container') || document.getElementById('alerts-list');
    if (!container) return;

    if (alerts.length === 0) {
        container.innerHTML = '<div class="no-alerts">No alerts found</div>';
        return;
    }

    let html = '';

    alerts.forEach(alert => {
        const time = new Date(alert.timestamp).toLocaleString();
        const status = alert.is_resolved ? 'Resolved' : 'Active';
        const statusClass = alert.is_resolved ? 'resolved' : 'active';

        html += `
            <div class="alert-item">
                <span class="alert-type">${alert.alert_type}</span>
                <span class="alert-severity ${alert.severity}">${alert.severity}</span>
                <span class="alert-time">${time}</span>
                <span class="alert-description">${alert.description}</span>
                <span class="alert-status ${statusClass}">${status}</span>
            </div>
        `;
    });

    container.innerHTML = html;
}

// ========================
// Update Alert Stats
// ========================
function updateAlertStats(alerts) {
    const activeAlerts = alerts.filter(alert => !alert.is_resolved).length;

    if (document.getElementById('active-alerts')) {
        document.getElementById('active-alerts').textContent = activeAlerts;
    }
}

// ========================
// Get Current Location
// ========================
function getCurrentLocation() {
    return new Promise((resolve, reject) => {
        if (!navigator.geolocation) {
            reject(new Error('Geolocation is not supported by this browser.'));
        }

        navigator.geolocation.getCurrentPosition(resolve, reject, {
            enableHighAccuracy: true,
            timeout: 5000,
            maximumAge: 0
        });
    });
}

// ========================
// Geolocation Tracking
// ========================
function startLocationTracking() {
    if (navigator.geolocation) {
        navigator.geolocation.watchPosition(
            (position) => {
                const { latitude, longitude } = position.coords;

                sendLocation(latitude, longitude);

                // Update map if available
                if (window.userMarker && window.map) {
                    window.userMarker.setLatLng([latitude, longitude]);
                    window.map.panTo([latitude, longitude]);
                }
            },
            (error) => console.error("Error getting location:", error),
            { enableHighAccuracy: true, maximumAge: 0 }
        );
    } else {
        alert("Geolocation is not supported by your browser.");
    }
}

// ========================
// Logout Function
// ========================
function handleLogout() {
    localStorage.removeItem('authToken');
    localStorage.removeItem('currentUser');
    window.location.href = '/login';
}

// ========================
// Initialization
// ========================
document.addEventListener("DOMContentLoaded", () => {
    // Check if user is logged in
    const token = localStorage.getItem('authToken');
    const currentUser = JSON.parse(localStorage.getItem('currentUser') || 'null');

    // Set up event listeners
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', (e) => {
            e.preventDefault();
            login();
        });
    }

    const registerForm = document.getElementById('register-form');
    if (registerForm) {
        registerForm.addEventListener('submit', (e) => {
            e.preventDefault();
            register();
        });
    }

    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }

    const panicBtn = document.getElementById('panic-btn');
    if (panicBtn) {
        panicBtn.addEventListener('click', sendPanicAlert);
    }

    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshAlerts);
    }

    const showResolvedCheckbox = document.getElementById('show-resolved');
    if (showResolvedCheckbox) {
        showResolvedCheckbox.addEventListener('change', refreshAlerts);
    }

    // If user is logged in and on dashboard, start location tracking
    if (token && currentUser && document.getElementById('username')) {
        document.getElementById('username').textContent = currentUser.username;
        startLocationTracking();
        refreshAlerts();

        // Set up periodic alerts refresh
        setInterval(refreshAlerts, 30000);
    }

    // Redirect to login if not authenticated and on protected page
    if (!token && !window.location.pathname.includes('/login') &&
        !window.location.pathname.includes('/register')) {
        window.location.href = '/login';
    }
});
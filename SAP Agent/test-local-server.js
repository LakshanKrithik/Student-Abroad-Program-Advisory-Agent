const axios = require('axios');

async function testLocalServer() {
    console.log('Attempting to connect to http://localhost:3000/api/create-web-call...');
    try {
        const response = await axios.get('http://localhost:3000/api/create-web-call');
        console.log('Server responded with status:', response.status);
        console.log('Response data:', response.data);

        if (response.data.access_token) {
            console.log('SUCCESS: Access Token received!');
        } else {
            console.log('FAILURE: No access token in response.');
        }
    } catch (error) {
        console.error('Connection failed:', error.message);
        if (error.response) {
            console.error('Server error details:', error.response.data);
        } else if (error.code === 'ECONNREFUSED') {
            console.error('ERROR: Could not connect to server. Is "node server.js" running?');
        }
    }
}

testLocalServer();

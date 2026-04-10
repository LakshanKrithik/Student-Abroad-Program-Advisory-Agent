const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const https = require('https');

dotenv.config();

const app = express();
const port = 3000;

app.use(cors());
app.use(express.json());

console.log('Starting server...');
console.log('API Key exists:', !!process.env.RETELL_API_KEY);
console.log('Agent ID exists:', !!process.env.RETELL_AGENT_ID);

app.get('/api/create-web-call', async (req, res) => {
  const agentId = process.env.RETELL_AGENT_ID;
  const apiKey = process.env.RETELL_API_KEY;

  if (!agentId) {
    return res.status(500).json({ error: 'RETELL_AGENT_ID not set in .env' });
  }

  if (!apiKey) {
    return res.status(500).json({ error: 'RETELL_API_KEY not set in .env' });
  }

  const body = JSON.stringify({ agent_id: agentId });

  const options = {
    hostname: 'api.retellai.com',
    path: '/v2/create-web-call',
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(body),
    },
  };

  try {
    const data = await new Promise((resolve, reject) => {
      const request = https.request(options, (response) => {
        let rawData = '';
        response.on('data', (chunk) => { rawData += chunk; });
        response.on('end', () => {
          try {
            const parsed = JSON.parse(rawData);
            if (response.statusCode >= 200 && response.statusCode < 300) {
              resolve(parsed);
            } else {
              reject(new Error(`Retell API error ${response.statusCode}: ${JSON.stringify(parsed)}`));
            }
          } catch (e) {
            reject(new Error(`Failed to parse response: ${rawData}`));
          }
        });
      });

      request.on('error', reject);
      request.write(body);
      request.end();
    });

    console.log('Web call created successfully:', data.call_id);
    res.json(data);

  } catch (error) {
    console.error('Error creating web call:', error.message);
    res.status(500).json({ error: 'Failed to create web call', details: error.message });
  }
});

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});

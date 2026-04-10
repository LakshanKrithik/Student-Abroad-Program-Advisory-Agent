const axios = require('axios');
require('dotenv').config();

async function test() {
    const apiKey = process.env.RETELL_API_KEY;
    const agentId = process.env.RETELL_AGENT_ID;

    console.log('Testing API with Agent ID:', agentId);

    try {
        const response = await axios.post(
            'https://api.retellai.com/v2/create-web-call',
            { agent_id: agentId },
            {
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json'
                }
            }
        );
        console.log('Success:', response.data);
    } catch (error) {
        console.error('Error with v2:', error.message);
        if (error.response) console.error('Data:', error.response.data);

        // Try v1 or alternative domain
        try {
            console.log('Trying alternative domain...');
            const response = await axios.post(
                'https://api.re-tell.ai/create-web-call',
                { agent_id: agentId },
                {
                    headers: {
                        'Authorization': `Bearer ${apiKey}`,
                        'Content-Type': 'application/json'
                    }
                }
            );
            console.log('Success with re-tell.ai:', response.data);
        } catch (err2) {
            console.error('Error with re-tell.ai:', err2.message);
            if (err2.response) console.error('Data:', err2.response.data);
        }
    }
}

test();

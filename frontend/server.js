/* eslint-disable no-undef */
// 后端代理服务器 - 安全地调用 DeepSeek API
import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const openai = new OpenAI({
  baseURL: 'https://api.deepseek.com',
  apiKey: process.env.DEEPSEEK_API_KEY
});

app.post('/api/ai-decision', async (req, res) => {
  try {
    const { prompt } = req.body;
    
    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }

    console.log('Received AI decision request');
    
    const completion = await openai.chat.completions.create({
      model: 'deepseek-chat',
      messages: [
        {
          role: 'system',
          content: 'You are an expert Texas Hold\'em poker AI. Always respond with valid JSON only, without markdown code blocks.'
        },
        {
          role: 'user',
          content: prompt
        }
      ],
      temperature: 0.7,
      max_tokens: 300
    });

    const content = completion.choices[0].message.content;
    console.log('AI response received');
    
    res.json({ content });
    
  } catch (error) {
    console.error('Error calling DeepSeek API:', error);
    res.status(500).json({ 
      error: 'Failed to get AI decision',
      message: error.message 
    });
  }
});


const PORT = process.env.PORT || 3001;

app.listen(PORT, () => {
  console.log(`Backend server running on http://localhost:${PORT}`);
});


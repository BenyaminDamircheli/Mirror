import { openai } from '@ai-sdk/openai'
import { streamText, generateText } from 'ai'
import fetch from 'node-fetch'

export async function POST(request: Request) {
    const {messages} = await request.json();
    const userMessage = messages[messages.length - 1].content;

    const apiUrl = "http://127.0.0.1:5000/search";
    const apiResponse = await fetch(apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: userMessage,
        }),
    });

    const {results} = await apiResponse.json();

    const topResults = Object.entries(results).slice(0, 3)

    const context = topResults.map(([filename, summary], index) => ({
        role:"system",
        content: `Filename: ${filename} Content: ${summary}`,
    }));

    console.log(context);
    
    const prompt = {
        model: openai("gpt-4o"),
        apiKey: process.env.OPENAI_API_KEY,
        messages: [
            ...messages,
            ...context,
            {
                role: "system",
                content: "Please directly answer the question concisely, unless specified by user, using the content, and cite it at the end of the sentence using the following format: '[citation number in increasing order from 1](filename of the source)'. It is EXTREMELY IMPORTANT that you cite your sources using the filename proveded to you (NOT THE ENTIRE FILEPATH). Do not EVER go on tangents. You are a helpful assistant with access to key parts and references into my life. Make inferences if needed. Always add a lot of detail and use at least one source. Do NOT add a list of references at the end.",
            }, 
            {
                role: "user",
                content: userMessage,
            }
        ],
    }

    const response = await streamText(prompt);
    return response.toAIStreamResponse();

}
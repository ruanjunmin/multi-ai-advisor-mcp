import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import fetch from "node-fetch";
import * as dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config();

// Ollama API URL from environment or fallback to default
const OLLAMA_API_URL = process.env.OLLAMA_API_URL || "http://localhost:11434";

// Specific models to use
const DEFAULT_MODELS = process.env.DEFAULT_MODELS?.split(",") || ["gemma3:1b", "llama3.2:1b", "deepseek-r1:1.5b"];

// Create server instance
const server = new McpServer({
  name: process.env.SERVER_NAME || "multi-model-advisor",
  version: process.env.SERVER_VERSION || "1.0.0",
});

// Define Ollama response types
interface OllamaResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
}

interface OllamaModel {
  name: string;
  modified_at: string;
  size: number;
  digest: string;
  details: {
    format: string;
    family: string;
    families: string[];
    parameter_size: string;
    quantization_level: string;
  };
}

// Fix the type for system prompts with index signature
interface SystemPrompts {
  [key: string]: string;
}

// Default system prompts for each model
const DEFAULT_SYSTEM_PROMPTS: SystemPrompts = {
  "gemma3:1b": process.env.GEMMA_SYSTEM_PROMPT || 
    "You are a creative and innovative AI assistant. Think outside the box and offer novel perspectives.",
  "llama3.2:1b": process.env.LLAMA_SYSTEM_PROMPT || 
    "You are a supportive and empathetic AI assistant focused on human well-being. Provide considerate and balanced advice.",
  "deepseek-r1:1.5b": process.env.DEEPSEEK_SYSTEM_PROMPT || 
    "You are a logical and analytical AI assistant. Think step-by-step and explain your reasoning clearly."
};

// Debug log if enabled
const debugLog = (message: string) => {
  if (process.env.DEBUG === "true") {
    console.error(`[DEBUG] ${message}`);
  }
};

// Tool to list available models in Ollama
server.tool(
  "list-available-models",
  "List all available models in Ollama that can be used with query-models",
  {},
  async () => {
    try {
      const response = await fetch(`${OLLAMA_API_URL}/api/tags`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json() as { models: OllamaModel[] };
      
      if (!data.models || !Array.isArray(data.models)) {
        return {
          content: [
            {
              type: "text",
              text: "No models found or unexpected response format from Ollama API."
            }
          ]
        };
      }
      
      // Format model information
      const modelInfo = data.models.map(model => {
        const size = (model.size / (1024 * 1024 * 1024)).toFixed(2); // Convert to GB
        const paramSize = model.details?.parameter_size || "Unknown";
        const quantLevel = model.details?.quantization_level || "Unknown";
        
        return `- **${model.name}**: ${paramSize} parameters, ${size} GB, ${quantLevel} quantization`;
      }).join("\n");
      
      // Show which models are currently configured as defaults
      const defaultModelsInfo = DEFAULT_MODELS.map(model => {
        const isAvailable = data.models.some(m => m.name === model);
        return `- **${model}**: ${isAvailable ? "✓ Available" : "⚠️ Not available"}`;
      }).join("\n");
      
      return {
        content: [
          {
            type: "text",
            text: `# Available Ollama Models\n\n${modelInfo}\n\n## Current Default Models\n\n${defaultModelsInfo}\n\nYou can use any of the available models with the query-models tool by specifying them in the 'models' parameter.`
          }
        ]
      };
    } catch (error) {
      console.error("Error listing models:", error);
      return {
        isError: true,
        content: [
          {
            type: "text",
            text: `Error listing models: ${error instanceof Error ? error.message : String(error)}\n\nMake sure Ollama is running and accessible at ${OLLAMA_API_URL}.`
          }
        ]
      };
    }
  }
);

// Register the tool for querying multiple models
server.tool(
  "query-models",
  "Query multiple AI models via Ollama and get their responses to compare perspectives",
  {
    question: z.string().describe("The question to ask all models"),
    models: z.array(z.string()).optional().describe("Array of model names to query (defaults to configured models)"),
    system_prompt: z.string().optional().describe("Optional system prompt to provide context to all models (overridden by model_system_prompts if provided)"),
    model_system_prompts: z.record(z.string()).optional().describe("Optional object mapping model names to specific system prompts"),
  },
  async ({ question, models, system_prompt, model_system_prompts }) => {
    try {
      // Use provided models or fall back to default models from environment
      const modelsToQuery = models || DEFAULT_MODELS;
      
      debugLog(`Using models: ${modelsToQuery.join(", ")}`);
      
      // Query each model in parallel
      const responses = await Promise.all(
        modelsToQuery.map(async (modelName) => {
          try {
            // Determine which system prompt to use for this model
            let modelSystemPrompt = system_prompt || "You are a helpful AI assistant answering a user's question.";
            
            // If model-specific prompts are provided, use those instead
            if (model_system_prompts && model_system_prompts[modelName]) {
              modelSystemPrompt = model_system_prompts[modelName];
            }
            // If no prompt is specified at all, use our default role-specific prompts if available
            else if (!system_prompt && modelName in DEFAULT_SYSTEM_PROMPTS) {
              modelSystemPrompt = DEFAULT_SYSTEM_PROMPTS[modelName];
            }

            debugLog(`Querying ${modelName} with system prompt: ${modelSystemPrompt.substring(0, 50)}...`);
            
            const response = await fetch(`${OLLAMA_API_URL}/api/generate`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                model: modelName,
                prompt: question,
                system: modelSystemPrompt,
                stream: false,
              }),
            });

            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json() as OllamaResponse;
            return {
              model: modelName,
              response: data.response,
              systemPrompt: modelSystemPrompt
            };
          } catch (modelError) {
            console.error(`Error querying model ${modelName}:`, modelError);
            return {
              model: modelName,
              response: `Error: Could not get response from ${modelName}. Make sure this model is available in Ollama.`,
              error: true
            };
          }
        })
      );

      // Format the response in a way that's easy for Claude to analyze
      const formattedText = `# Responses from Multiple Models\n\n${responses.map(resp => {
        const roleInfo = resp.systemPrompt ? 
          `*Role: ${resp.systemPrompt.substring(0, 100)}${resp.systemPrompt.length > 100 ? '...' : ''}*\n\n` : '';
        
        return `## ${resp.model.toUpperCase()} RESPONSE:\n${roleInfo}${resp.response}\n\n`;
      }).join("")}\n\nConsider the perspectives above when formulating your response. You may agree or disagree with any of these models. Note that these are all compact 1-1.5B parameter models, so take that into account when evaluating their responses.`;

      return {
        content: [
          {
            type: "text",
            text: formattedText,
          },
        ],
      };
    } catch (error) {
      console.error("Error in query-models tool:", error);
      return {
        isError: true,
        content: [
          {
            type: "text",
            text: `Error querying models: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
      };
    }
  }
);

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`Multi-Model Advisor MCP Server running on stdio`);
  console.error(`Using Ollama API URL: ${OLLAMA_API_URL}`);
  console.error(`Default models: ${DEFAULT_MODELS.join(", ")}`);
  
  if (process.env.DEBUG === "true") {
    console.error("Debug mode enabled");
    console.error("Default system prompts:");
    Object.entries(DEFAULT_SYSTEM_PROMPTS).forEach(([model, prompt]) => {
      console.error(`${model}: ${prompt.substring(0, 50)}...`);
    });
  }
}

main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});
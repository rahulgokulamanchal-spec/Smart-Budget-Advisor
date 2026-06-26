import express from "express";
import path from "path";
import dotenv from "dotenv";
import { GoogleGenAI, Type } from "@google/genai";
import { createServer as createViteServer } from "vite";

dotenv.config();

// Helper to retry calls to the Gemini API with dynamic fallback to high-availability models in case of 503/429
async function callWithRetryAndFallback<T>(
  fn: (modelName: string) => Promise<T>
): Promise<T> {
  const models = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest"
  ];
  let lastError: any = null;

  for (const model of models) {
    let attempts = 2;
    let delay = 500;
    while (attempts > 0) {
      try {
        return await fn(model);
      } catch (error: any) {
        lastError = error;
        const errorMessage = error?.message || "";
        const errorString = typeof error === 'object' ? JSON.stringify(error) : String(error);
        
        const isNotFound = 
          errorString.includes("404") ||
          errorString.includes("not_found") ||
          errorMessage.includes("not found") ||
          errorMessage.includes("model");

        const isRetryable = 
          errorString.includes("503") ||
          errorString.includes("429") ||
          errorString.includes("UNAVAILABLE") ||
          errorMessage.includes("demand") ||
          errorMessage.includes("temporary") ||
          errorMessage.includes("temporarily") ||
          errorString.includes("ResourceExhausted") ||
          errorString.includes("Service Unavailable") ||
          errorString.includes("overloaded");

        if (isNotFound) {
          console.warn(`Gemini API (${model}) is not available (404/not found). Moving to next fallback model...`, errorMessage);
          break; // Break the inner while loop to move to the next model in the models array
        } else if (isRetryable && attempts > 1) {
          console.warn(`Gemini API (${model}) failed with retryable error. Retrying same model in ${delay}ms... (Attempts left: ${attempts - 1})`, errorMessage);
          await new Promise(resolve => setTimeout(resolve, delay));
          attempts--;
          delay *= 1.5;
        } else if (isRetryable) {
          console.warn(`Gemini API (${model}) failed/exhausted. Moving to next fallback model...`, errorMessage);
          break; // Break the inner while loop to move to the next model in the models array
        } else {
          // Non-retryable error (e.g. invalid argument, bad request), throw immediately
          throw error;
        }
      }
    }
  }

  throw lastError || new Error("All Gemini models failed to respond");
}

// Helper to retry calls to the Claude (Anthropic) API with dynamic fallback to high-availability models in case of 503/429
async function callClaudeWithRetryAndFallback(
  prompt: string,
  systemInstruction: string,
  jsonMode = false
): Promise<string> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey || apiKey === "MY_ANTHROPIC_API_KEY") {
    throw new Error("ANTHROPIC_API_KEY_MISSING");
  }

  // Fallback list of models
  const models = [
    "claude-3-5-sonnet-latest",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-latest",
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307"
  ];
  let lastError: any = null;

  for (const model of models) {
    let attempts = 2;
    let delay = 500;
    while (attempts > 0) {
      try {
        const url = "https://api.anthropic.com/v1/messages";
        const headers = {
          "x-api-key": apiKey,
          "anthropic-version": "2023-06-01",
          "content-type": "application/json",
        };

        const messages: any[] = [
          { role: "user", content: prompt }
        ];

        if (jsonMode) {
          messages.push({ role: "assistant", content: "{" });
        }

        const body = {
          model,
          max_tokens: 4000,
          system: systemInstruction,
          messages,
        };

        const response = await fetch(url, {
          method: "POST",
          headers,
          body: JSON.stringify(body),
        });

        if (!response.ok) {
          const errText = await response.text();
          throw new Error(`Claude API ${model} failed with status ${response.status}: ${errText}`);
        }

        const result: any = await response.json();
        let contentText = result.content?.[0]?.text || "";
        if (jsonMode) {
          contentText = "{" + contentText;
        }
        return contentText;
      } catch (error: any) {
        lastError = error;
        const errorMessage = error?.message || "";
        const errorString = typeof error === 'object' ? JSON.stringify(error) : String(error);
        
        const isNotFound = 
          errorString.includes("404") ||
          errorString.includes("not_found") ||
          errorMessage.includes("not found") ||
          errorMessage.includes("model:");

        const isRetryable = 
          errorString.includes("503") ||
          errorString.includes("429") ||
          errorString.includes("UNAVAILABLE") ||
          errorMessage.includes("demand") ||
          errorMessage.includes("temporary") ||
          errorMessage.includes("temporarily") ||
          errorString.includes("ResourceExhausted") ||
          errorString.includes("Service Unavailable") ||
          errorString.includes("overloaded");

        if (isNotFound) {
          console.warn(`Claude API (${model}) is not available (404/not found). Moving to next fallback model...`, errorMessage);
          break; // Move to next model
        } else if (isRetryable && attempts > 1) {
          console.warn(`Claude API (${model}) failed with retryable error. Retrying same model in ${delay}ms... (Attempts left: ${attempts - 1})`, errorMessage);
          await new Promise((resolve) => setTimeout(resolve, delay));
          attempts--;
          delay *= 1.5;
        } else if (isRetryable) {
          console.warn(`Claude API (${model}) failed/exhausted. Moving to next fallback model...`, errorMessage);
          break; // Move to next model
        } else {
          // Non-retryable error, throw immediately
          throw error;
        }
      }
    }
  }

  throw lastError || new Error("All Claude models failed to respond");
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // Lazy initialize Gemini client or return null if key is missing
  const getGeminiClient = () => {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey || apiKey === "MY_GEMINI_API_KEY") {
      return null;
    }
    return new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  };

  const getAnthropicKey = () => {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey || apiKey === "MY_ANTHROPIC_API_KEY") {
      return null;
    }
    return apiKey;
  };

  // 1. Initial structured budget advice endpoint
  app.post("/api/budget-advice", async (req, res) => {
    try {
      const anthropicKey = getAnthropicKey();
      const ai = getGeminiClient();

      if (!anthropicKey && !ai) {
        return res.status(200).json({
          error: "API_KEY_MISSING",
          message: "Please configure your ANTHROPIC_API_KEY or GEMINI_API_KEY in the Settings > Secrets panel of Google AI Studio to get professional AI insights."
        });
      }

      const { income, expenses, customCategoryExpenses, financialGoal } = req.body;

      // Prepare a readable representation of the data for the AI
      const expenseSummary = Object.entries(expenses)
        .map(([cat, amt]) => `- ${cat}: $${amt}`)
        .join("\n");

      const customSummary = customCategoryExpenses && customCategoryExpenses.length > 0
        ? customCategoryExpenses.map((c: { name: string; amount: number }) => `- ${c.name}: $${c.amount}`).join("\n")
        : "None";

      const totalRegularExpenses = Object.values(expenses).reduce((sum: number, val: any) => sum + (Number(val) || 0), 0);
      const totalCustomExpenses = customCategoryExpenses
        ? customCategoryExpenses.reduce((sum: number, val: any) => sum + (Number(val.amount) || 0), 0)
        : 0;
      const totalExpenses = totalRegularExpenses + totalCustomExpenses;
      const netSavings = income - totalExpenses;

      const prompt = `Analyze this monthly budget for a personal finance assistant.
Income: $${income}
Standard Expenses:
${expenseSummary}
Custom Expenses:
${customSummary}
Total Monthly Expenses: $${totalExpenses}
Net Savings: $${netSavings}
User's Financial Goal: ${financialGoal || "General budget optimization and saving advice."}

Evaluate their spending proportions. For reference, mention the 50/30/20 rule (50% essentials, 30% wants, 20% savings) or other relevant standards. Provide clear recommendations.`;

      let responseText = "";
      let usedGeminiFallback = false;

      if (anthropicKey) {
        try {
          console.log("Analyzing budget using Claude API...");
          const systemInstruction = `You are an expert, encouraging, and highly analytical AI Financial Planner. Analyze the user's monthly budget. Calculate ratios of spending (such as necessities, wants, and savings). Score their budget health from 0 to 100. Pinpoint categories of potential overspending and suggest actionable, specific improvements.

IMPORTANT: You MUST respond ONLY with a valid JSON object matching this schema:
{
  "healthScore": number, // 0 to 100
  "healthRating": string, // "Poor" | "Fair" | "Good" | "Excellent"
  "ratios": {
    "necessities": number, // percentage spent on essentials (e.g. rent, food, transport)
    "wants": number, // percentage spent on wants (e.g. entertainment, custom hobby expenses)
    "savings": number // percentage saved out of total income (can be negative if running deficit)
  },
  "summary": "A cohesive, friendly, and structured overview of their budget health and overall standing.",
  "recommendations": [
    {
      "category": string, // "rent" | "food" | "transport" | "entertainment" | "savings" etc.
      "severity": "high" | "medium" | "low",
      "title": string,
      "description": string
    }
  ],
  "categoryBreakdownAdvice": {
    "rent": string,
    "food": string,
    "transport": string,
    "entertainment": string
  }
}

Do not include any other markdown formatting (like backticks or \`\`\`json) outside of the JSON object. Since the assistant message is prefilled with "{", you just complete the JSON object.`;

          responseText = await callClaudeWithRetryAndFallback(prompt, systemInstruction, true);
        } catch (claudeError: any) {
          console.warn("Claude API failed, trying to fall back to Gemini API...", claudeError);
          if (ai) {
            usedGeminiFallback = true;
          } else {
            throw claudeError;
          }
        }
      }

      if (!responseText && (ai || usedGeminiFallback)) {
        console.log("Analyzing budget using Gemini API...");
        const response = await callWithRetryAndFallback((modelName) => ai!.models.generateContent({
          model: modelName,
          contents: prompt,
          config: {
            systemInstruction: "You are an expert, encouraging, and highly analytical AI Financial Planner. Analyze the user's monthly budget. Calculate ratios of spending (such as necessities, wants, and savings). Score their budget health from 0 to 100. Pinpoint categories of potential overspending and suggest actionable, specific improvements.",
            responseMimeType: "application/json",
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                healthScore: {
                  type: Type.INTEGER,
                  description: "A financial health score from 0 to 100 based on their budget breakdown and savings rate."
                },
                healthRating: {
                  type: Type.STRING,
                  description: "Rating category: e.g. Poor, Fair, Good, Excellent."
                },
                ratios: {
                  type: Type.OBJECT,
                  description: "Analysis of spending percentages compared to income",
                  properties: {
                    necessities: { type: Type.NUMBER, description: "Percentage spent on essentials (e.g. rent, standard food, transport)" },
                    wants: { type: Type.NUMBER, description: "Percentage spent on wants/discretionary (e.g. entertainment, custom hobby expenses)" },
                    savings: { type: Type.NUMBER, description: "Percentage saved out of total income (or negative if running a deficit)" }
                  },
                  required: ["necessities", "wants", "savings"]
                },
                summary: {
                  type: Type.STRING,
                  description: "A cohesive, friendly, and structured overview of their budget health and overall standing."
                },
                recommendations: {
                  type: Type.ARRAY,
                  description: "Actionable tips prioritized by impact.",
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      category: { type: Type.STRING, description: "Relevant category name, e.g. rent, food, transport, entertainment, or savings." },
                      severity: { type: Type.STRING, description: "Priority level: high, medium, or low." },
                      title: { type: Type.STRING, description: "A concise title of the advice." },
                      description: { type: Type.STRING, description: "A detailed description explaining how to achieve this saving or optimization." }
                    },
                    required: ["category", "severity", "title", "description"]
                  }
                },
                categoryBreakdownAdvice: {
                  type: Type.OBJECT,
                  description: "Short specific feedback on each category",
                  properties: {
                    rent: { type: Type.STRING },
                    food: { type: Type.STRING },
                    transport: { type: Type.STRING },
                    entertainment: { type: Type.STRING }
                  },
                  required: ["rent", "food", "transport", "entertainment"]
                }
              },
              required: [
                "healthScore",
                "healthRating",
                "ratios",
                "summary",
                "recommendations",
                "categoryBreakdownAdvice"
              ]
            }
          }
        }));

        if (!response.text) {
          throw new Error("No response text from Gemini API");
        }
        responseText = response.text;
      }

      const budgetAnalysis = JSON.parse(responseText.trim());
      res.json({ success: true, analysis: budgetAnalysis });
    } catch (error: any) {
      console.error("Error in /api/budget-advice:", error);
      const errorString = typeof error === 'object' ? JSON.stringify(error) : String(error);
      const errorMessage = error?.message || "";
      
      if (
        errorString.includes("503") ||
        errorString.includes("UNAVAILABLE") ||
        errorMessage.includes("demand") ||
        errorMessage.includes("temporary") ||
        errorMessage.includes("temporarily") ||
        errorString.includes("Service Unavailable") ||
        errorString.includes("overloaded") ||
        errorString.includes("429") ||
        errorString.includes("RESOURCE_EXHAUSTED") ||
        errorString.includes("quota")
      ) {
        return res.status(200).json({
          success: false,
          error: "The AI service is currently experiencing very high demand or quota limits (429/503). Please wait a few seconds and click 'Get AI Budgeting Advice' to try again."
        });
      }
      res.status(200).json({ success: false, error: error.message || "Failed to analyze budget" });
    }
  });

  // 2. Chat endpoint for follow-up budget questions
  app.post("/api/chat", async (req, res) => {
    try {
      const anthropicKey = getAnthropicKey();
      const ai = getGeminiClient();

      if (!anthropicKey && !ai) {
        return res.status(200).json({
          error: "API_KEY_MISSING",
          message: "Please configure your ANTHROPIC_API_KEY or GEMINI_API_KEY in Settings."
        });
      }

      const { messages, budgetData } = req.body;

      if (!messages || !Array.isArray(messages)) {
        return res.status(400).json({ error: "Messages array is required" });
      }

      // Initialize chat session with background budget info
      const systemInstruction = `You are an expert personal finance coach.
The user is discussing their monthly budget:
- Monthly Income: $${budgetData.income}
- Rent/Housing: $${budgetData.expenses.rent}
- Food: $${budgetData.expenses.food}
- Transport: $${budgetData.expenses.transport}
- Entertainment: $${budgetData.expenses.entertainment}
${budgetData.customCategoryExpenses && budgetData.customCategoryExpenses.length > 0 
  ? `- Custom Expenses: ${budgetData.customCategoryExpenses.map((c: any) => `${c.name}: $${c.amount}`).join(", ")}`
  : ""
}

Be encouraging, specific, and professional. Provide practical, creative tips to cut costs, increase income, or build emergency funds. Answer direct financial planning questions accurately but recommend seeking certified professionals for specific legal/tax questions. Keep your responses relatively concise (under 200 words) and formatted beautifully in markdown.`;

      // Let's structure the prompt history clearly:
      let formattedPrompt = "This is a continuation of the budget coaching conversation. Here is our history:\n\n";
      messages.forEach((msg: any) => {
        formattedPrompt += `${msg.role === "user" ? "User" : "Assistant"}: ${msg.content}\n\n`;
      });
      formattedPrompt += "Assistant: [Provide your response here]";

      let responseText = "";
      let usedGeminiFallback = false;

      if (anthropicKey) {
        try {
          console.log("Processing chat with Claude API...");
          responseText = await callClaudeWithRetryAndFallback(formattedPrompt, systemInstruction, false);
        } catch (claudeError: any) {
          console.warn("Claude API chat failed, trying to fall back to Gemini API...", claudeError);
          if (ai) {
            usedGeminiFallback = true;
          } else {
            throw claudeError;
          }
        }
      }

      if (!responseText && (ai || usedGeminiFallback)) {
        console.log("Processing chat with Gemini API...");
        const response = await callWithRetryAndFallback((modelName) => ai!.models.generateContent({
          model: modelName,
          contents: formattedPrompt,
          config: {
            systemInstruction
          }
        }));
        if (!response.text) {
          throw new Error("No response text from Gemini API");
        }
        responseText = response.text;
      }

      res.json({ success: true, text: responseText });
    } catch (error: any) {
      console.error("Error in /api/chat:", error);
      const errorString = typeof error === 'object' ? JSON.stringify(error) : String(error);
      const errorMessage = error?.message || "";
      
      if (
        errorString.includes("503") ||
        errorString.includes("UNAVAILABLE") ||
        errorMessage.includes("demand") ||
        errorMessage.includes("temporary") ||
        errorMessage.includes("temporarily") ||
        errorString.includes("Service Unavailable") ||
        errorString.includes("overloaded") ||
        errorString.includes("429") ||
        errorString.includes("RESOURCE_EXHAUSTED") ||
        errorString.includes("quota")
      ) {
        return res.status(200).json({
          success: false,
          error: "The AI service is currently experiencing very high demand or quota limits (429/503). Please wait a few seconds and send your message again."
        });
      }
      res.status(200).json({ success: false, error: error.message || "Failed to process chat" });
    }
  });

  // Vite middleware for development, or static serving for production
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on port ${PORT}`);
  });
}

startServer();

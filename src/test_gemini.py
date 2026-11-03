
from ai_agent import ask_gemini

print("\n====================================")
print("       PARALILOOP AI AGENT")
print("====================================\n")

question = "What are three alternatives to stubble burning in Punjab?"

print("Farmer:", question)
print("\nConnecting to Gemini...\n")

answer = ask_gemini(question)

print("ParaliLoop AI:")
print(answer)

print("\n====================================")

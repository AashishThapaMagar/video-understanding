import json
import ollama

# Load the timeline we built earlier
with open("timeline.json") as f:
    timeline = json.load(f)

# Turn the timeline into readable text for the model
timeline_text = "\n".join(
    f"At {e['second']}s: " + ", ".join(f"{v} {k}" for k, v in e.items() if k != "second")
    for e in timeline
)

print("=== Soccer Video Q&A ===")
print("Here's what was detected in the video:\n")
print(timeline_text)
print("\nAsk a question about the video (or type 'quit' to stop).\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ("quit", "exit", ""):
        print("Goodbye!")
        break

    prompt = f"""You are analyzing a soccer video. Below is what an object detector found, second by second.
Each line shows how many of each object were visible ON SCREEN at that second — these are simultaneous counts, NOT new or unique objects. Never add the per-second numbers together. The largest count on a single line is the most of that object seen at one time.

{timeline_text}

Answer the user's question in 1-3 sentences, based only on this data. If the data doesn't contain the answer, say so honestly.

Question: {question}"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
    )
    print("\nAI:", response["message"]["content"], "\n")
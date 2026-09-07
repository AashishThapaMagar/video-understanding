import json
import os
import ollama

# Load the timeline we built earlier
with open("timeline.json") as f:
    timeline = json.load(f)

# Turn the timeline into readable text for the model
timeline_text = "\n".join(
    f"At {e['second']}s: " + ", ".join(f"{v} {k}" for k, v in e.items() if k != "second")
    for e in timeline
)

# Load the pass/shot counts from events.py, if that has been run
events_text = ""
if os.path.exists("events.json"):
    with open("events.json") as f:
        report = json.load(f)

    totals = report["totals"]
    lines = [f"Totals: {totals['passes']} passes, {totals['turnovers']} turnovers, "
             f"{totals['shots']} shots.", "", "Per player (IDs are tracking IDs, not shirt numbers):"]
    for p in report["players"]:
        if not (p["passes_attempted"] or p["passes_received"] or p["shots"]):
            continue
        lines.append(
            f"  Player {p['id']} (team {p['team']}, {p['role']}): "
            f"{p['passes_completed']} passes completed, {p['turnovers']} lost, "
            f"{p['passes_received']} received, {p['shots']} shots, "
            f"{p['possession_seconds']}s on the ball"
        )

    lines += ["", "Events in order:"]
    for e in report["events"]:
        if e["type"] == "shot":
            lines.append(f"  {e['second']}s: player {e['from']} (team {e['team']}) had a shot")
        else:
            verb = "passed to" if e["type"] == "pass" else "lost the ball to"
            lines.append(f"  {e['second']}s: player {e['from']} {verb} player {e['to']}")

    events_text = "\n".join(lines)

print("=== Soccer Video Q&A ===")
print("Here's what was detected in the video:\n")
print(timeline_text)
if events_text:
    print("\n--- Passes and shots ---")
    print(events_text)
else:
    print("\n(no events.json yet — run 'python events.py' for pass and shot counts)")
print("\nAsk a question about the video (or type 'quit' to stop).\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ("quit", "exit", ""):
        print("Goodbye!")
        break

    prompt = f"""You are analyzing a soccer video. Below is what an object detector found, second by second.
Each line shows how many of each object were visible ON SCREEN at that second — these are simultaneous counts, NOT new or unique objects. Never add the per-second numbers together. The largest count on a single line is the most of that object seen at one time.

{timeline_text}
"""

    if events_text:
        prompt += f"""
Here are the passes and shots worked out by tracking who was nearest the ball. Unlike the per-second counts above, these ARE cumulative totals for the whole clip, so they can be added up and compared. Player numbers are tracking IDs, not shirt numbers, and the same person may appear under more than one ID if they left the frame. A "shot" means a fast ball heading towards a goalkeeper or the goal end of the frame.

{events_text}
"""

    prompt += f"""
Answer the user's question in 1-3 sentences, based only on this data. If the data doesn't contain the answer, say so honestly.

Question: {question}"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
    )
    print("\nAI:", response["message"]["content"], "\n")

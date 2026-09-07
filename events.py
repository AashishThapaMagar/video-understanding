"""Count passes and shots per player in a soccer clip.

How it works:
1. Track people with YOLO + BoT-SORT (see soccer_tracker.yaml), so a player keeps the
   same ID across frames even while the camera pans.
2. Link ball detections into ball tracks — a clip can contain several balls at once,
   so the tracks are built here rather than left to the object tracker.
3. Decide who is "on the ball" each frame — the nearest player within reach of it.
4. Turn changes of holder into events: pass, turnover (ball ends up with the other
   team) and shot (ball leaves fast and heads at a goalkeeper / the goal end).
5. Save per-player pass and shot counts to events.json for the Q&A step.

Run it with:
    python events.py                # counts only
    python events.py --save-video   # plus events_annotated.mp4 to check the calls

Everything here happens in frame coordinates, with no pitch calibration and no
jersey-number reading, so treat the numbers as good estimates rather than
official match stats. The constants below are the knobs worth tuning.
"""

from ultralytics import YOLO
import argparse
import json
import math
import cv2
import numpy as np
from collections import defaultdict, Counter

WEIGHTS = r"C:\Users\Aashish\runs\detect\train-6\weights\best.pt"
VIDEO = "soccer.mp4"
TRACKER = "soccer_tracker.yaml"   # BoT-SORT tuned for a shaky, panning camera

# --- ball linking ---
BALL_MIN_CONF = 0.20      # ignore very weak ball detections
BALL_GATE = 0.05          # a ball may move this fraction of the frame width per frame
BALL_MAX_GAP = 12         # keep a ball track alive across this many missed frames
BALL_MIN_LENGTH = 6       # drop ball tracks shorter than this (usually false positives)

# --- possession ---
POSSESSION_REACH = 0.75   # ball counts as "held" within this * the player's box height
MIN_HOLD_FRAMES = 3       # ignore touches shorter than this (jitter)
MERGE_GAP_FRAMES = 8      # same player re-touching within this gap is still one possession

# --- events ---
# Distances are fractions of the frame width, speeds are fractions of the frame width
# per frame, so the thresholds survive a change of resolution.
PASS_MAX_SECONDS = 2.5    # ball must reach the next player within this to count as a pass
MIN_PASS_TRAVEL = 0.025   # the ball has to actually travel, not just change owner
MIN_PASS_SPEED = 0.004    # ...and travel at some pace: rules out a ball sitting still
MIN_PASS_SEPARATION = 1.0 # passer and receiver must be this many box-heights apart,
                          # which throws out ID swaps on one and the same person
SHOT_MIN_SPEED = 0.015    # a struck ball is much faster than a rolled pass
SHOT_MIN_TRAVEL = 0.15    # ...and covers real ground
SHOT_WINDOW_SECONDS = 1.5 # how far ahead to watch the ball after the last touch
GK_REACH = 3.0            # ball ending this * the keeper's box height away counts as at goal
GOAL_BAND = 0.10          # outer band of the frame treated as the goal end

# --- roles ---
# The detector flips between "player", "goalkeeper" and "referee" on the same person
# from frame to frame, so each role is decided once per track instead.
GK_ROLE_RATIO = 0.15      # this share of goalkeeper frames makes the whole track a keeper
GK_MIN_FRAMES = 5         # ...but never off the back of two or three stray frames
REF_ROLE_RATIO = 0.50     # referees have to be called a referee most of the time

# --- team colours ---
JERSEY_MAX_SAMPLES = 25   # colour samples kept per player
JERSEY_MIN_SAMPLES = 3    # players with fewer samples get no team
TEAM_SEPARATION = 1.10    # below this the two colour clusters are not convincing

BALL_TRAIL = 12           # how many past ball positions to draw on the annotated video

BOX_COLORS = {
    "A": (235, 170, 60),      # blue-ish
    "B": (70, 140, 250),      # orange-ish
    "?": (180, 180, 180),
    "referee": (60, 220, 240),
    "ball": (255, 255, 255),
}


# ---------------------------------------------------------------- geometry

def point_box_distance(px, py, box):
    """Distance from a point to a box (0 when the point is inside it)."""
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def runs_of(seq):
    """Group a sequence into [value, start_index, end_index] runs."""
    grouped = []
    for i, value in enumerate(seq):
        if grouped and grouped[-1][0] == value:
            grouped[-1][2] = i
        else:
            grouped.append([value, i, i])
    return grouped


# ---------------------------------------------------------------- pass 1: detect + track

def jersey_color(frame, box):
    """Median Lab colour of a player's shirt, with pitch-green pixels thrown away."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    box_w, box_h = x2 - x1, y2 - y1
    if box_w < 8 or box_h < 20:
        return None

    # torso only: skip the head and the shorts/legs, and trim the sides
    ty1 = max(0, y1 + int(0.15 * box_h))
    ty2 = min(height, y1 + int(0.50 * box_h))
    tx1 = max(0, x1 + int(0.20 * box_w))
    tx2 = min(width, x2 - int(0.20 * box_w))
    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    is_grass = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 60)
    keep = ~is_grass
    if keep.sum() < 20:
        keep = np.ones_like(is_grass)

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)[keep.reshape(-1)]
    return np.median(lab.astype(np.float32), axis=0)


def track_video(model, video_path, conf, tracker):
    """Run detection + tracking once, collecting people, balls and shirt colours."""
    people_by_frame = defaultdict(list)
    balls_by_frame = defaultdict(list)
    colors_by_id = defaultdict(list)
    classes_by_id = defaultdict(Counter)
    frame_count = 0

    results = model.track(video_path, stream=True, conf=conf,
                          tracker=tracker, verbose=False)

    for frame_index, r in enumerate(results):
        frame_count = frame_index + 1
        ids = r.boxes.id
        ids = ids.int().cpu().tolist() if ids is not None else [None] * len(r.boxes)

        for box, track_id in zip(r.boxes, ids):
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

            if label == "ball":
                if confidence >= BALL_MIN_CONF:
                    balls_by_frame[frame_index].append(
                        {"pos": ((x1 + x2) / 2, (y1 + y2) / 2), "conf": confidence}
                    )
                continue

            if track_id is None:
                continue

            people_by_frame[frame_index].append(
                {"id": track_id, "label": label, "box": (x1, y1, x2, y2)}
            )
            classes_by_id[track_id][label] += 1

            if label in ("player", "goalkeeper") and len(colors_by_id[track_id]) < JERSEY_MAX_SAMPLES:
                color = jersey_color(r.orig_img, (x1, y1, x2, y2))
                if color is not None:
                    colors_by_id[track_id].append(color)

        if frame_index % 100 == 0:
            print(f"  ...tracked {frame_index} frames")

    return people_by_frame, balls_by_frame, colors_by_id, classes_by_id, frame_count


# ---------------------------------------------------------------- roles

def resolve_roles(classes_by_id):
    """One role per track, from all its frames — a keeper called 'player' half the time
    is still a keeper, and that matters for both possession and shot detection."""
    roles = {}
    for track_id, counts in classes_by_id.items():
        total = sum(counts.values())
        if not total:
            continue
        if counts["goalkeeper"] >= GK_MIN_FRAMES and counts["goalkeeper"] / total >= GK_ROLE_RATIO:
            roles[track_id] = "goalkeeper"
        elif counts["referee"] / total >= REF_ROLE_RATIO:
            roles[track_id] = "referee"
        else:
            roles[track_id] = "player"
    return roles


# ---------------------------------------------------------------- teams

def assign_teams(colors_by_id, roles):
    """Split shirt colours into two teams with k-means. Returns {track_id: 'A'/'B'/'?'}."""
    outfield, keepers = {}, {}
    for track_id, samples in colors_by_id.items():
        if len(samples) < JERSEY_MIN_SAMPLES or roles.get(track_id) == "referee":
            continue
        median = np.median(np.stack(samples), axis=0)
        (keepers if roles.get(track_id) == "goalkeeper" else outfield)[track_id] = median

    teams = defaultdict(lambda: "?")
    if len(outfield) < 4:
        print("  (too few players to tell the teams apart — everyone counts as one team)")
        for track_id in list(outfield) + list(keepers):
            teams[track_id] = "A"
        return teams, None

    ids = list(outfield)
    data = np.stack([outfield[i] for i in ids]).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()

    # Are the two clusters actually different kits, or just one team split in half?
    spread = np.mean([np.linalg.norm(data[i] - centers[labels[i]]) for i in range(len(ids))])
    separation = float(np.linalg.norm(centers[0] - centers[1]) / max(spread, 1e-6))
    if separation < TEAM_SEPARATION:
        print(f"  (kit colours too similar, separation {separation:.2f} — counting one team)")
        for track_id in list(outfield) + list(keepers):
            teams[track_id] = "A"
        return teams, separation

    # the bigger cluster becomes team A, so the labels stay stable between runs
    big = 0 if (labels == 0).sum() >= (labels == 1).sum() else 1
    names = {big: "A", 1 - big: "B"}
    for track_id, cluster in zip(ids, labels):
        teams[track_id] = names[int(cluster)]

    # keepers wear their own kit, so this is only a best guess
    for track_id, color in keepers.items():
        nearest = int(np.argmin([np.linalg.norm(color - c) for c in centers]))
        teams[track_id] = names[nearest]

    print(f"  two kits found (separation {separation:.2f})")
    return teams, separation


# ---------------------------------------------------------------- ball tracklets

def build_ball_tracks(balls_by_frame, frame_count, frame_width):
    """Link ball detections frame to frame into tracklets, bridging short dropouts."""
    gate = BALL_GATE * frame_width
    active, finished = [], []

    for frame_index in range(frame_count):
        detections = balls_by_frame.get(frame_index, [])

        candidates = []
        for ti, track in enumerate(active):
            gap = frame_index - track["frames"][-1]
            predicted = track["pos"][-1]
            if len(track["pos"]) >= 2:
                vx = track["pos"][-1][0] - track["pos"][-2][0]
                vy = track["pos"][-1][1] - track["pos"][-2][1]
                predicted = (predicted[0] + vx, predicted[1] + vy)
            for di, det in enumerate(detections):
                distance = math.dist(predicted, det["pos"])
                if distance <= gate * gap:      # the gate widens while the ball is missing
                    candidates.append((distance, ti, di))

        candidates.sort()
        matched_tracks, matched_dets = set(), set()
        for _, ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            track, det = active[ti], detections[di]

            last_frame, last_pos = track["frames"][-1], track["pos"][-1]
            for missed in range(last_frame + 1, frame_index):   # fill the dropout
                t = (missed - last_frame) / (frame_index - last_frame)
                track["frames"].append(missed)
                track["pos"].append((last_pos[0] + t * (det["pos"][0] - last_pos[0]),
                                     last_pos[1] + t * (det["pos"][1] - last_pos[1])))
            track["frames"].append(frame_index)
            track["pos"].append(det["pos"])

        for di, det in enumerate(detections):
            if di not in matched_dets:
                active.append({"frames": [frame_index], "pos": [det["pos"]]})

        still_active = []
        for track in active:
            (still_active if frame_index - track["frames"][-1] <= BALL_MAX_GAP
             else finished).append(track)
        active = still_active

    finished += active
    return [t for t in finished if len(t["frames"]) >= BALL_MIN_LENGTH]


# ---------------------------------------------------------------- possession

def holder_per_frame(track, people_by_frame, roles):
    """For each step of a ball track, the ID of the player within reach (or None)."""
    holders = []
    for frame_index, (bx, by) in zip(track["frames"], track["pos"]):
        best_id, best_distance = None, None
        for person in people_by_frame.get(frame_index, []):
            if roles.get(person["id"]) == "referee":
                continue        # referees don't take possession
            x1, y1, x2, y2 = person["box"]
            distance = point_box_distance(bx, by, person["box"])
            if distance <= POSSESSION_REACH * (y2 - y1):
                if best_distance is None or distance < best_distance:
                    best_id, best_distance = person["id"], distance
        holders.append(best_id)
    return holders


def possession_segments(holders):
    """Collapse per-frame holders into [player_id, start_index, end_index] segments."""
    cleaned = list(holders)
    for value, start, end in runs_of(holders):
        if value is not None and (end - start + 1) < MIN_HOLD_FRAMES:
            for i in range(start, end + 1):
                cleaned[i] = None

    segments = []
    for value, start, end in runs_of(cleaned):
        if value is None:
            continue
        if segments and segments[-1][0] == value and start - segments[-1][2] <= MERGE_GAP_FRAMES:
            segments[-1][2] = end
        else:
            segments.append([value, start, end])
    return segments


# ---------------------------------------------------------------- events

def flight_stats(track, start, end, frame_width):
    """How far and how fast the ball moved between two touches (both as fractions
    of the frame width, so they don't depend on the video's resolution)."""
    positions = track["pos"][start:end + 1]
    frames = track["frames"][start:end + 1]
    if len(positions) < 2:
        return 0.0, 0.0
    travelled = sum(math.dist(positions[i], positions[i + 1]) for i in range(len(positions) - 1))
    speed = travelled / max(frames[-1] - frames[0], 1)
    return travelled / frame_width, speed / frame_width


def box_of(people_by_frame, frame_index, track_id):
    for person in people_by_frame.get(frame_index, []):
        if person["id"] == track_id:
            return person["box"]
    return None


def players_apart(passer_box, receiver_box):
    """Gap between two people, measured in box-heights so distance from the camera
    doesn't matter. Two boxes on top of each other mean one person, not a pass."""
    if passer_box is None or receiver_box is None:
        return None
    ax = (passer_box[0] + passer_box[2]) / 2, (passer_box[1] + passer_box[3]) / 2
    bx = (receiver_box[0] + receiver_box[2]) / 2, (receiver_box[1] + receiver_box[3]) / 2
    mean_height = ((passer_box[3] - passer_box[1]) + (receiver_box[3] - receiver_box[1])) / 2
    return math.dist(ax, bx) / max(mean_height, 1e-6)


def looks_like_shot(track, start, end, people_by_frame, roles, frame_width, travel, speed):
    """Fast ball covering ground towards a keeper or the goal end of the frame."""
    if end - start < 2 or speed < SHOT_MIN_SPEED or travel < SHOT_MIN_TRAVEL:
        return False

    positions = track["pos"][start:end + 1]
    frames = track["frames"][start:end + 1]
    end_x, end_y = positions[-1]

    # ended at a goalkeeper (saved, caught, or beat them)
    for frame_index in frames[-5:]:
        for person in people_by_frame.get(frame_index, []):
            if roles.get(person["id"]) != "goalkeeper":
                continue
            x1, y1, x2, y2 = person["box"]
            if point_box_distance(end_x, end_y, person["box"]) <= GK_REACH * (y2 - y1):
                return True

    # or ran off the goal end of the frame, still moving that way
    moving_left = positions[-1][0] < positions[0][0]
    if moving_left and end_x < GOAL_BAND * frame_width:
        return True
    if not moving_left and end_x > (1 - GOAL_BAND) * frame_width:
        return True
    return False


def events_for_ball(track, ball_index, people_by_frame, teams, roles, fps, frame_width):
    """Read one ball track's possession changes as passes, turnovers and shots."""
    segments = possession_segments(holder_per_frame(track, people_by_frame, roles))
    pass_max_frames = PASS_MAX_SECONDS * fps
    shot_window = int(SHOT_WINDOW_SECONDS * fps)
    events = []

    for i, (holder, _, end) in enumerate(segments):
        following = segments[i + 1] if i + 1 < len(segments) else None
        if following:
            receiver, receive_start = following[0], following[1]
            flight_end = receive_start
            gap_frames = track["frames"][receive_start] - track["frames"][end]
        else:
            receiver, gap_frames = None, None
            flight_end = min(end + shot_window, len(track["frames"]) - 1)

        # The pass test looks at the whole flight from one touch to the next; the shot
        # test only looks at the moment after the strike, because a ball that is picked
        # up again ten seconds later would otherwise average out to a slow roll.
        travel, speed = flight_stats(track, end, flight_end, frame_width)
        shot_end = min(flight_end, end + shot_window)
        shot_travel, shot_speed = flight_stats(track, end, shot_end, frame_width)

        event = {
            "frame": track["frames"][end],
            "second": round(track["frames"][end] / fps, 1),
            "from": holder,
            "to": receiver,
            "team": teams[holder],
            "ball_track": ball_index,
            "flight": [end, flight_end],   # step range inside that ball track
            "travel": round(travel, 3),
            "speed": round(speed, 4),
        }

        if looks_like_shot(track, end, shot_end, people_by_frame, roles,
                           frame_width, shot_travel, shot_speed):
            event["type"] = "shot"
            event["flight"] = [end, shot_end]
            event["travel"], event["speed"] = round(shot_travel, 3), round(shot_speed, 4)
            if receiver is not None and gap_frames is not None and gap_frames <= pass_max_frames:
                event["ended_with"] = receiver
            event["to"] = None
            events.append(event)
            continue

        if receiver is None or receiver == holder or gap_frames > pass_max_frames:
            continue        # dribble, or the ball just ran loose

        # A change of owner is only a pass if the ball really travelled between two
        # separate people — otherwise it is a tracker wobble around a still ball.
        separation = players_apart(box_of(people_by_frame, track["frames"][end], holder),
                                   box_of(people_by_frame, track["frames"][receive_start], receiver))
        if (travel < MIN_PASS_TRAVEL or speed < MIN_PASS_SPEED
                or separation is None or separation < MIN_PASS_SEPARATION):
            continue

        same_team = teams[holder] == teams[receiver] and teams[holder] != "?"
        event["type"] = "pass" if same_team else "turnover"
        event["to_team"] = teams[receiver]
        event["separation"] = round(separation, 1)
        events.append(event)

    return events, segments


# ---------------------------------------------------------------- stats

def build_stats(events, ball_tracks, all_segments, teams, roles, fps):
    stats = defaultdict(lambda: {"passes_completed": 0, "passes_attempted": 0,
                                 "passes_received": 0, "turnovers": 0, "shots": 0,
                                 "possession_seconds": 0.0})

    for ball_index, segments in all_segments.items():
        track = ball_tracks[ball_index]
        for holder, start, end in segments:
            held = track["frames"][end] - track["frames"][start] + 1
            stats[holder]["possession_seconds"] += held / fps

    for event in events:
        passer = stats[event["from"]]
        if event["type"] == "shot":
            passer["shots"] += 1
        elif event["type"] == "pass":
            passer["passes_completed"] += 1
            passer["passes_attempted"] += 1
            stats[event["to"]]["passes_received"] += 1
        elif event["type"] == "turnover":
            passer["turnovers"] += 1
            passer["passes_attempted"] += 1

    players = []
    for track_id, row in stats.items():
        players.append({
            "id": track_id,
            "team": teams[track_id],
            "role": roles.get(track_id, "player"),
            "possession_seconds": round(row["possession_seconds"], 1),
            **{k: v for k, v in row.items() if k != "possession_seconds"},
        })
    players.sort(key=lambda p: (p["passes_completed"] + p["shots"], p["possession_seconds"]),
                 reverse=True)
    return players


# ---------------------------------------------------------------- annotated video

def draw_text_bar(frame, text, y, color, scale):
    """Text on a dark strip, so it stays readable over grass, crowd or a scoreboard."""
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, max(2, int(2.5 * scale))
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, pad = 20, int(10 * scale)
    y1, y2 = max(0, y - text_h - pad), min(frame.shape[0], y + pad)
    x2 = min(frame.shape[1], x + text_w + 2 * pad)
    strip = frame[y1:y2, max(0, x - pad):x2]
    if strip.size:
        frame[y1:y2, max(0, x - pad):x2] = cv2.addWeighted(strip, 0.35,
                                                           np.zeros_like(strip), 0.65, 0)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def save_annotated_video(video_path, out_path, people_by_frame, ball_tracks,
                         all_segments, events, teams, roles, fps):
    """Second pass over the video, drawing IDs, teams, possession and events."""
    holder_at = defaultdict(set)
    for ball_index, segments in all_segments.items():
        track = ball_tracks[ball_index]
        for holder, start, end in segments:
            for step in range(start, end + 1):
                holder_at[track["frames"][step]].add(holder)

    # every ball, plus the few positions before it, so a struck ball leaves a trail
    ball_at = defaultdict(list)
    for track in ball_tracks:
        for step, (frame_index, pos) in enumerate(zip(track["frames"], track["pos"])):
            trail = track["pos"][max(0, step - BALL_TRAIL):step + 1]
            ball_at[frame_index].append((pos, trail))

    banner_frames = int(1.2 * fps)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    scale = max(width, height) / 1400.0

    frame_index = 0
    passes = shots = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for person in people_by_frame.get(frame_index, []):
            x1, y1, x2, y2 = [int(v) for v in person["box"]]
            role = roles.get(person["id"], "player")
            if role == "referee":
                color, tag = BOX_COLORS["referee"], "REF"
            else:
                team = teams[person["id"]]
                color = BOX_COLORS[team]
                tag = f"#{person['id']}" + (" GK" if role == "goalkeeper" else "")
                if team in ("A", "B"):
                    tag += f" {team}"
            on_ball = person["id"] in holder_at.get(frame_index, ())
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if on_ball else 1)
            cv2.putText(frame, tag, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * scale, color, max(1, int(1.5 * scale)), cv2.LINE_AA)

        for (bx, by), trail in ball_at.get(frame_index, []):
            for i in range(len(trail) - 1):
                cv2.line(frame, (int(trail[i][0]), int(trail[i][1])),
                         (int(trail[i + 1][0]), int(trail[i + 1][1])),
                         BOX_COLORS["ball"], max(1, int(1.5 * scale)), cv2.LINE_AA)
            cv2.circle(frame, (int(bx), int(by)), int(7 * scale), BOX_COLORS["ball"], 2)

        for event in events:
            if 0 <= frame_index - event["frame"] < banner_frames:
                if event["type"] == "shot":
                    text, highlight = f"SHOT  #{event['from']}", (60, 60, 255)
                elif event["type"] == "pass":
                    text, highlight = f"PASS  #{event['from']} -> #{event['to']}", (90, 255, 90)
                else:
                    text, highlight = f"LOST  #{event['from']} -> #{event['to']}", (60, 200, 255)

                # redraw the ball's actual path for this event, so you can see the call
                flight = ball_tracks[event["ball_track"]]["pos"][event["flight"][0]:event["flight"][1] + 1]
                for i in range(len(flight) - 1):
                    cv2.line(frame, (int(flight[i][0]), int(flight[i][1])),
                             (int(flight[i + 1][0]), int(flight[i + 1][1])),
                             highlight, max(2, int(3 * scale)), cv2.LINE_AA)

                draw_text_bar(frame, text, int(height - 40 * scale), highlight, 1.0 * scale)
                break

        passes = sum(1 for e in events if e["type"] == "pass" and e["frame"] <= frame_index)
        shots = sum(1 for e in events if e["type"] == "shot" and e["frame"] <= frame_index)
        # sits below the broadcast scoreboard that is burned into the top of the frame
        draw_text_bar(frame, f"passes {passes}   shots {shots}", int(0.10 * height),
                      (255, 255, 255), 0.9 * scale)

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Per-player pass and shot counts for a soccer clip")
    parser.add_argument("--video", default=VIDEO)
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--tracker", default=TRACKER)
    parser.add_argument("--out", default="events.json")
    parser.add_argument("--save-video", action="store_true",
                        help="also write events_annotated.mp4 with IDs and events drawn on")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    model = YOLO(args.weights)

    print("Tracking players and the ball...")
    people_by_frame, balls_by_frame, colors_by_id, classes_by_id, frame_count = track_video(
        model, args.video, args.conf, args.tracker
    )

    roles = resolve_roles(classes_by_id)
    keepers = sum(1 for r in roles.values() if r == "goalkeeper")
    print(f"  {len(roles)} tracked people ({keepers} goalkeeper(s))")

    print("Working out the teams from shirt colours...")
    teams, separation = assign_teams(colors_by_id, roles)

    print("Linking ball detections into ball tracks...")
    ball_tracks = build_ball_tracks(balls_by_frame, frame_count, frame_width)
    print(f"  {len(ball_tracks)} ball track(s)")

    print("Reading possession changes as passes and shots...")
    events, all_segments = [], {}
    for ball_index, track in enumerate(ball_tracks):
        found, segments = events_for_ball(track, ball_index, people_by_frame,
                                          teams, roles, fps, frame_width)
        events += found
        all_segments[ball_index] = segments
    events.sort(key=lambda e: e["frame"])

    players = build_stats(events, ball_tracks, all_segments, teams, roles, fps)

    totals = Counter(e["type"] for e in events)
    report = {
        "video": args.video,
        "fps": fps,
        "duration_seconds": round(frame_count / fps, 1),
        "totals": {
            "passes": totals["pass"],
            "turnovers": totals["turnover"],
            "shots": totals["shot"],
        },
        "teams_detected": sorted({t for t in teams.values() if t != "?"}),
        "kit_separation": round(separation, 2) if separation else None,
        "note": ("Counts come from tracking who is nearest each ball. Player IDs are "
                 "tracker IDs, not shirt numbers, and a player who leaves the frame and "
                 "comes back may get a new ID. A 'shot' is a fast ball heading at a "
                 "goalkeeper or the goal end of the frame."),
        "players": players,
        "events": events,
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- Passes and shots ---")
    print(f"{totals['pass']} passes, {totals['turnover']} turnovers, {totals['shot']} shots\n")
    print(f"{'player':>8}  {'team':>4}  {'role':>10}  {'passes':>6}  {'lost':>4}  "
          f"{'recv':>4}  {'shots':>5}  {'on ball':>7}")
    for player in players[:20]:
        print(f"{'#' + str(player['id']):>8}  {player['team']:>4}  {player['role']:>10}  "
              f"{player['passes_completed']:>6}  {player['turnovers']:>4}  "
              f"{player['passes_received']:>4}  {player['shots']:>5}  "
              f"{player['possession_seconds']:>6.1f}s")

    print("\n--- Event timeline ---")
    for event in events:
        if event["type"] == "shot":
            print(f"{event['second']:>5.1f}s  SHOT by #{event['from']} (team {event['team']})")
        else:
            word = "passes to" if event["type"] == "pass" else "loses it to"
            print(f"{event['second']:>5.1f}s  #{event['from']} {word} #{event['to']}")

    print(f"\nSaved {len(events)} events to {args.out}")

    if args.save_video:
        print("Drawing the annotated video...")
        save_annotated_video(args.video, "events_annotated.mp4", people_by_frame,
                             ball_tracks, all_segments, events, teams, roles, fps)
        print("Saved events_annotated.mp4")


if __name__ == "__main__":
    main()

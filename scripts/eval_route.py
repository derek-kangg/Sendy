"""Evaluate AI beta generation against a climber-verified beta.

Usage:
    python scripts/eval_route.py test_routes/route-01-black-slab

Loads the route's holds + verified beta, runs the sequence engine
headlessly (real API calls), and scores the AI output against the
verified beta. Results are printed and saved to eval_result.json.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
from dotenv import load_dotenv
from PIL import Image

from route_graph import build_reachability_graph
from sequence_engine import build_holds_description, generate_sequence_iteratively

HANDS = ("left hand", "right hand", "both hands")


def hand_moves(moves):
    """Ordered (limb, hold) hand placements, including matches/finish."""
    out = []
    for m in moves:
        if m["limb"] in HANDS and m.get("hold") is not None:
            out.append((m["limb"], m["hold"]))
    return out


def foot_moves(moves):
    out = []
    for m in moves:
        if m["limb"] not in HANDS:
            out.append((m["limb"], m.get("hold")))
    return out


def lcs(a, b):
    """Length of the longest common subsequence."""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def fmt_moves(moves):
    lines = []
    for i, m in enumerate(moves, 1):
        hold = f"hold {m['hold']}" if m.get("hold") is not None else "wall"
        note = m.get("note") or m.get("cue") or ""
        lines.append(f"  {i:2d}. {m['limb']:<16} -> {hold:<8} {note}")
    return "\n".join(lines)


def main(route_dir):
    load_dotenv()
    with open(os.path.join(route_dir, "route_data.json"), encoding="utf-8") as f:
        data = json.load(f)

    holds       = data["holds"]
    verified    = data["verified_beta"]["moves"]
    start_holds = data["start_holds"]
    height_cm   = data.get("height_cm", 170)

    img = Image.open(os.path.join(route_dir, "wall.jpg"))
    image_width, image_height = img.size

    graph = build_reachability_graph(holds, image_height, image_width, height_cm)
    hold_descriptions = build_holds_description(holds)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    print(f"Generating beta for {route_dir} ({data['difficulty']}, {data['wall_angle']})...")
    t0 = time.time()

    def progress(n, total):
        print(f"  move {n}...", end="\r")

    result = generate_sequence_iteratively(
        client, hold_descriptions, holds, graph, start_holds,
        height_cm, data["difficulty"], data["wall_angle"],
        data.get("finish_style", "TOP label"), "",
        image_height=image_height, progress_callback=progress,
    )
    elapsed = time.time() - t0
    sequence = result.sequence
    ai_moves = [m for m in sequence if m["move_number"] > 0]
    print(f"\nStatus: {result.status} ({result.calls} API calls)"
          + (f" — {result.reason}" if result.reason else ""))

    # ---- scoring ----
    v_hands, a_hands = hand_moves(verified), hand_moves(ai_moves)
    v_feet,  a_feet  = foot_moves(verified), foot_moves(ai_moves)

    hand_pair_score = lcs(v_hands, a_hands) / len(v_hands) if v_hands else 0
    v_hand_holds = [h for _, h in v_hands]
    a_hand_holds = [h for _, h in a_hands]
    hand_hold_score = lcs(v_hand_holds, a_hand_holds) / len(v_hand_holds) if v_hand_holds else 0
    foot_pair_score = lcs(v_feet, a_feet) / len(v_feet) if v_feet else 0

    print(f"\nDone in {elapsed:.0f}s — {len(ai_moves)} AI moves vs {len(verified)} verified moves\n")
    print("VERIFIED BETA (climber):")
    print(fmt_moves(verified))
    print("\nAI BETA:")
    print(fmt_moves(ai_moves))
    print("\nSCORES:")
    print(f"  Hand sequence (limb+hold):  {hand_pair_score:.0%}   ({lcs(v_hands, a_hands)}/{len(v_hands)} in order)")
    print(f"  Hand hold order (any limb): {hand_hold_score:.0%}   ({lcs(v_hand_holds, a_hand_holds)}/{len(v_hand_holds)} in order)")
    print(f"  Foot moves (limb+hold):     {foot_pair_score:.0%}   ({lcs(v_feet, a_feet)}/{len(v_feet)} in order)")
    print(f"  Move count: AI {len(ai_moves)} vs verified {len(verified)}")

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(elapsed, 1),
        "ai_sequence": ai_moves,
        "scores": {
            "hand_pair": round(hand_pair_score, 3),
            "hand_hold_order": round(hand_hold_score, 3),
            "foot_pair": round(foot_pair_score, 3),
            "ai_moves": len(ai_moves),
            "verified_moves": len(verified),
        },
    }
    with open(os.path.join(route_dir, "eval_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {os.path.join(route_dir, 'eval_result.json')}")


if __name__ == "__main__":
    route = sys.argv[1] if len(sys.argv) > 1 else "test_routes/route-01-black-slab"
    main(route)

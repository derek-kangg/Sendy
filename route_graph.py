DEFAULT_WALL_HEIGHT_CM = 400  # a typical bouldering wall, photographed top to bottom


def build_reachability_graph(holds, image_height, image_width, climber_height_cm,
                             wall_height_cm=DEFAULT_WALL_HEIGHT_CM):
    """
    Build a graph of reachable holds based on climber height and image scale.

    Scale comes from assuming the photo spans `wall_height_cm` of wall
    (default 4 m); the UI lets the climber correct it for short or tall walls.
    Arm span is approximately equal to height, max reach is height * 1.3.
    """
    pixels_per_cm = image_height / float(wall_height_cm or DEFAULT_WALL_HEIGHT_CM)
    
    max_reach_cm = climber_height_cm * 1.3
    max_reach_px = max_reach_cm * pixels_per_cm
    
    # Max horizontal reach is roughly arm span / 2 = height / 2
    max_horizontal_cm = climber_height_cm * 0.7
    max_horizontal_px = max_horizontal_cm * pixels_per_cm

    graph = {}

    for hold in holds:
        reachable = []
        for other in holds:
            if other["number"] == hold["number"]:
                continue

            dx = abs(other["x"] - hold["x"])
            dy = other["y"] - hold["y"]  # positive means lower on wall

            # (dy negative means other hold is higher up)
            euclidean = (dx**2 + dy**2) ** 0.5

            if euclidean <= max_reach_px and dx <= max_horizontal_px:
                # Calculate difficulty based on distance
                ratio = euclidean / max_reach_px
                if ratio < 0.4:
                    difficulty = "Easy"
                elif ratio < 0.7:
                    difficulty = "Medium"
                else:
                    difficulty = "Hard"

                reachable.append({
                    "hold": other["number"],
                    "distance_px": round(euclidean),
                    "difficulty": difficulty,
                    "direction": "up" if dy < 0 else "down or same level"
                })

        # Sort by how high up the wall (lower y = higher up)
        reachable.sort(key=lambda h: next(
            hold2["y"] for hold2 in holds if hold2["number"] == h["hold"]
        ))

        graph[hold["number"]] = reachable

    return graph


def format_graph_for_prompt(graph, holds, start_hold_numbers):
    """
    Convert the graph into a readable format for the AI prompt.
    """
    hold_map = {h["number"]: h for h in holds}
    
    lines = ["REACHABILITY GRAPH — which holds can be reached from each hold:"]
    lines.append(f"Start holds: {start_hold_numbers}")
    lines.append("")
    
    for hold_num, reachable in graph.items():
        hold = hold_map[hold_num]
        if reachable:
            reachable_str = ", ".join([
                f"Hold {r['hold']} ({r['difficulty']}, {r['direction']})"
                for r in reachable
            ])
        else:
            reachable_str = "none in reach"
        lines.append(f"Hold {hold_num} (x={hold['x']}, y={hold['y']}, size={hold.get('size','?')}): can reach → {reachable_str}")
    
    return "\n".join(lines)
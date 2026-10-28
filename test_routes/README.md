# Test Routes

Real climbs with known-good beta, used to evaluate and improve Sendy's
route generation. One folder per route.

## Folder format

```
test_routes/
  route-01-movement-sfu-blue-v2/
    wall.jpg          <- clear, straight-on photo of the route
    route.md          <- route info + your beta (template below)
```

## route.md template

```markdown
# Route 01 — Movement SFU, blue, V2

- Gym / wall: Movement SFU, slab wall
- Colour: Blue
- Grade: V2
- Wall angle: Slab
- Start: both hands on the big jug low left
- Finish: TOP-labelled hold

## My beta (verified by climbing it)

START: both hands hold X, LF hold Y, RF hold Z
1. RF steps to hold A
2. RH to hold B (half crimp)
3. ...

## Notes

Anything tricky: where people fall, alternative beta for shorter
climbers, holds that look better than they are, etc.
```

## What makes a good test route

- A range of grades (V0-V2 easy footwork, V3-V5 technique like drop
  knees and heel hooks, V6+ if available)
- A range of angles: slab, vertical, overhang
- At least one route with an undercling or side-pull, one with a
  mandatory foot swap or flag, one with a big move (deadpoint/dyno)
- Beta you have personally climbed and confirmed, not guessed

The more precisely the beta is written (which limb, which hold number,
which grip), the more useful it is for scoring the AI's output against it.

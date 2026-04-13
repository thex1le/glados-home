"""Analyze tracking.jsonl recording for GLaDOS head camera session."""
import json
import sys
from collections import Counter, defaultdict

JSONL_PATH = r"C:\Users\Ben\PycharmProjects\glados-home\recordings\session_20260412_175514\camera_head\tracking.jsonl"

def main():
    frames = []
    with open(JSONL_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))

    print(f"Total frames: {len(frames)}")
    print(f"\n{'='*80}")
    print("FIRST LINE (complete data structure):")
    print(json.dumps(frames[0], indent=2))

    # Extract all person detections per frame
    frame_data = []
    all_detections = []
    non_person_detections = []

    for fr in frames:
        fn = fr["frame_number"]
        tracking = fr.get("tracking", {})
        persons = []
        # Check all keys in tracking for detections
        for key, val in tracking.items():
            if key in ("trace_id", "ts_vision"):
                continue
            if isinstance(val, dict) and "objects" in val:
                for obj in val["objects"]:
                    if obj.get("name") == "person" or obj.get("class") == 0:
                        persons.append(obj)
                        all_detections.append((fn, obj))
                    else:
                        non_person_detections.append((fn, obj))

        frame_data.append({
            "frame": fn,
            "wall_time": fr.get("wall_time", 0),
            "persons": persons,
            "person_count": len(persons),
        })

    # 1. Person count distribution
    print(f"\n{'='*80}")
    print("1. PERSON COUNT DISTRIBUTION")
    count_dist = Counter(fd["person_count"] for fd in frame_data)
    for k in sorted(count_dist.keys()):
        pct = count_dist[k] / len(frame_data) * 100
        bar = "#" * int(pct / 2)
        print(f"  {k} persons: {count_dist[k]:4d} frames ({pct:5.1f}%) {bar}")

    # 2. Confidence histogram
    print(f"\n{'='*80}")
    print("2. CONFIDENCE HISTOGRAM")
    buckets = {"0.00-0.30": 0, "0.30-0.50": 0, "0.50-0.70": 0, "0.70-0.90": 0, "0.90-1.00": 0}
    below_040 = 0
    confs = []
    for fn, obj in all_detections:
        c = obj["confidence"]
        confs.append(c)
        if c < 0.30:
            buckets["0.00-0.30"] += 1
        elif c < 0.50:
            buckets["0.30-0.50"] += 1
        elif c < 0.70:
            buckets["0.50-0.70"] += 1
        elif c < 0.90:
            buckets["0.70-0.90"] += 1
        else:
            buckets["0.90-1.00"] += 1
        if c < 0.40:
            below_040 += 1

    for label, count in buckets.items():
        pct = count / max(len(confs), 1) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label}: {count:4d} ({pct:5.1f}%) {bar}")
    print(f"\n  Detections below 0.40 confidence floor: {below_040} / {len(confs)} ({below_040/max(len(confs),1)*100:.1f}%)")
    if confs:
        print(f"  Min confidence: {min(confs):.5f}")
        print(f"  Max confidence: {max(confs):.5f}")
        print(f"  Mean confidence: {sum(confs)/len(confs):.5f}")

    # 3. Track ID analysis
    print(f"\n{'='*80}")
    print("3. TRACK ID ANALYSIS")
    track_id_counts = Counter()
    for fn, obj in all_detections:
        tid = obj.get("track_id", "none")
        track_id_counts[tid] += 1
    print(f"  Unique track_ids: {len(track_id_counts)}")
    for tid, cnt in track_id_counts.most_common():
        print(f"    track_id={tid}: {cnt} detections")

    # Track ID changes for highest-confidence detection per frame
    prev_tid = None
    tid_changes = 0
    tid_change_frames = []
    for fd in frame_data:
        if fd["persons"]:
            best = max(fd["persons"], key=lambda p: p["confidence"])
            tid = best.get("track_id", "none")
            if prev_tid is not None and tid != prev_tid:
                tid_changes += 1
                tid_change_frames.append(fd["frame"])
            prev_tid = tid
        else:
            prev_tid = None

    print(f"\n  Track ID switches (best detection per frame): {tid_changes}")
    if tid_change_frames:
        print(f"  Switch frames: {tid_change_frames[:30]}{'...' if len(tid_change_frames) > 30 else ''}")

    # 4. Bounding box center trajectory
    print(f"\n{'='*80}")
    print("4. BOUNDING BOX CENTER TRAJECTORY (highest-confidence per frame)")
    prev_cx, prev_cy = None, None
    large_jumps = []  # > 50px
    huge_jumps = []   # > 100px

    for fd in frame_data:
        if fd["persons"]:
            best = max(fd["persons"], key=lambda p: p["confidence"])
            box = best["box"]
            cx = (box["x1"] + box["x2"]) / 2
            cy = (box["y1"] + box["y2"]) / 2
            if prev_cx is not None:
                dx = cx - prev_cx
                dy = cy - prev_cy
                dist = (dx**2 + dy**2) ** 0.5
                if dist > 50:
                    large_jumps.append((fd["frame"], dist, prev_cx, prev_cy, cx, cy))
                if dist > 100:
                    huge_jumps.append((fd["frame"], dist))
            prev_cx, prev_cy = cx, cy
        else:
            prev_cx, prev_cy = None, None

    print(f"  Jumps > 50px: {len(large_jumps)}")
    print(f"  Jumps > 100px: {len(huge_jumps)}")
    if large_jumps:
        print(f"\n  Largest jumps (frame, distance, from->to):")
        for frame, dist, pcx, pcy, cx, cy in sorted(large_jumps, key=lambda x: -x[1])[:15]:
            print(f"    Frame {frame:4d}: {dist:6.1f}px  ({pcx:.0f},{pcy:.0f}) -> ({cx:.0f},{cy:.0f})")

    # 5. Face ID analysis
    print(f"\n{'='*80}")
    print("5. FACE ID ANALYSIS")
    face_ids_seen = Counter()
    face_id_per_frame = []
    for fn, obj in all_detections:
        fid = obj.get("face_id", None)
        if fid is not None:
            face_ids_seen[fid] += 1
    if face_ids_seen:
        print(f"  Unique face_ids: {len(face_ids_seen)}")
        for fid, cnt in face_ids_seen.most_common():
            print(f"    '{fid}': {cnt} detections")

        # Face ID flicker analysis (best detection per frame)
        prev_fid = None
        fid_flickers = 0
        fid_flicker_frames = []
        for fd in frame_data:
            if fd["persons"]:
                best = max(fd["persons"], key=lambda p: p["confidence"])
                fid = best.get("face_id", None)
                if fid is not None and prev_fid is not None and fid != prev_fid:
                    fid_flickers += 1
                    fid_flicker_frames.append((fd["frame"], prev_fid, fid))
                prev_fid = fid
            else:
                prev_fid = None
        print(f"\n  Face ID flickers (best det per frame): {fid_flickers}")
        if fid_flicker_frames:
            print(f"  First 20 flickers:")
            for frame, old, new in fid_flicker_frames[:20]:
                print(f"    Frame {frame}: '{old}' -> '{new}'")
    else:
        print("  No face_id field found in detections")

    # 6. Multi-person frame analysis
    print(f"\n{'='*80}")
    print("6. MULTI-PERSON FRAME ANALYSIS")
    multi_frames = [fd for fd in frame_data if fd["person_count"] >= 2]
    print(f"  Frames with 2+ persons: {len(multi_frames)} / {len(frame_data)} ({len(multi_frames)/len(frame_data)*100:.1f}%)")
    if multi_frames:
        print(f"\n  Confidence pairs in multi-person frames (first 20):")
        for fd in multi_frames[:20]:
            confs_str = ", ".join(f"{p['confidence']:.3f} (tid={p.get('track_id','?')})" for p in sorted(fd["persons"], key=lambda x: -x["confidence"]))
            print(f"    Frame {fd['frame']:4d}: [{confs_str}]")

        # Check for phantoms: detections with very low confidence alongside high confidence
        phantom_count = 0
        for fd in multi_frames:
            cs = [p["confidence"] for p in fd["persons"]]
            if max(cs) > 0.6 and min(cs) < 0.35:
                phantom_count += 1
        print(f"\n  Likely phantom frames (high+low confidence mix): {phantom_count}")

    # 7. Detection dropout timeline
    print(f"\n{'='*80}")
    print("7. DETECTION DROPOUT TIMELINE (person_count = 0)")
    dropout_ranges = []
    in_dropout = False
    start_frame = None
    for fd in frame_data:
        if fd["person_count"] == 0:
            if not in_dropout:
                in_dropout = True
                start_frame = fd["frame"]
        else:
            if in_dropout:
                dropout_ranges.append((start_frame, fd["frame"] - 1))
                in_dropout = False
    if in_dropout:
        dropout_ranges.append((start_frame, frame_data[-1]["frame"]))

    print(f"  Total dropout ranges: {len(dropout_ranges)}")
    total_dropout_frames = sum(e - s + 1 for s, e in dropout_ranges)
    print(f"  Total dropout frames: {total_dropout_frames} / {len(frame_data)} ({total_dropout_frames/len(frame_data)*100:.1f}%)")
    if dropout_ranges:
        print(f"\n  Dropout ranges (start-end, length):")
        for s, e in dropout_ranges:
            length = e - s + 1
            print(f"    Frames {s:4d}-{e:4d} ({length} frames)")

    # 8. Non-person detections
    print(f"\n{'='*80}")
    print("8. NON-PERSON DETECTIONS")
    if non_person_detections:
        np_classes = Counter(obj.get("name", f"class_{obj.get('class', '?')}") for _, obj in non_person_detections)
        print(f"  Total non-person detections: {len(non_person_detections)}")
        for cls, cnt in np_classes.most_common():
            print(f"    '{cls}': {cnt}")
        print(f"\n  First 10 non-person detections:")
        for fn, obj in non_person_detections[:10]:
            print(f"    Frame {fn}: {obj.get('name', '?')} conf={obj['confidence']:.3f}")
    else:
        print("  None found -- no phantom object classes!")

    # 9. Frame-by-frame state changes
    print(f"\n{'='*80}")
    print("9. DRAMATIC STATE CHANGES")

    dramatic = []
    prev_count = None
    prev_best_cx, prev_best_cy = None, None
    prev_best_conf = None

    for fd in frame_data:
        fn = fd["frame"]
        cnt = fd["person_count"]
        events = []

        # Count jumps
        if prev_count is not None:
            if (prev_count == 0 and cnt >= 2) or (prev_count >= 2 and cnt == 0):
                events.append(f"person count {prev_count}->{cnt}")

        # Best detection analysis
        if fd["persons"]:
            best = max(fd["persons"], key=lambda p: p["confidence"])
            box = best["box"]
            cx = (box["x1"] + box["x2"]) / 2
            cy = (box["y1"] + box["y2"]) / 2
            conf = best["confidence"]

            if prev_best_cx is not None:
                dist = ((cx - prev_best_cx)**2 + (cy - prev_best_cy)**2) ** 0.5
                if dist > 100:
                    events.append(f"bbox jump {dist:.0f}px")

            if prev_best_conf is not None and prev_best_conf > 0.7 and conf < 0.4:
                events.append(f"confidence drop {prev_best_conf:.2f}->{conf:.2f}")

            prev_best_cx, prev_best_cy = cx, cy
            prev_best_conf = conf
        else:
            prev_best_cx, prev_best_cy = None, None
            prev_best_conf = None

        if events:
            dramatic.append((fn, events))
        prev_count = cnt

    print(f"  Total dramatic events: {len(dramatic)}")
    if dramatic:
        for fn, events in dramatic:
            print(f"    Frame {fn:4d}: {', '.join(events)}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"  Recording: {len(frames)} frames")
    if frame_data:
        t0 = frame_data[0]["wall_time"]
        t1 = frame_data[-1]["wall_time"]
        print(f"  Duration: {t1 - t0:.1f}s")
        print(f"  FPS: {len(frames) / max(t1 - t0, 0.001):.1f}")
    print(f"  Total person detections: {len(all_detections)}")
    print(f"  Unique track IDs: {len(track_id_counts)}")
    print(f"  Frames with 0 persons: {count_dist.get(0, 0)} ({count_dist.get(0, 0)/len(frame_data)*100:.1f}%)")
    print(f"  Frames with 2+ persons: {len(multi_frames)} ({len(multi_frames)/len(frame_data)*100:.1f}%)")
    print(f"  Detections below 0.40 floor: {below_040}")
    print(f"  Non-person detections: {len(non_person_detections)}")
    print(f"  Large bbox jumps (>50px): {len(large_jumps)}")
    print(f"  Track ID switches: {tid_changes}")
    print(f"  Dropout ranges: {len(dropout_ranges)} ({total_dropout_frames} frames)")
    print(f"  Dramatic state changes: {len(dramatic)}")

if __name__ == "__main__":
    main()

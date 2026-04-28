import csv
import cv2
import numpy as np
from pathlib import Path
from collections import Counter, deque


# =========================
# CONFIGURATION
# =========================
BASE_DIR = Path(__file__).resolve().parent

RAW_BOARD_DIR = BASE_DIR / "King Domino dataset"
TILES_DIR = BASE_DIR / "DATASETT PROCESSED"
MODEL_PATH = BASE_DIR / "models" / "terrain_knn.npz"
BOARD_SCORES_PATH = BASE_DIR / "board_scores.csv"

TEMPLATE_PATHS = [
    BASE_DIR / "Crown Templates" / "krone_mork.jpeg",
    BASE_DIR / "Crown Templates" / "krone_lys.jpeg",
    BASE_DIR / "Crown Templates" / "krone_utydelig.jpeg",
    BASE_DIR / "Crown Templates" / "krone_gul.png",
]

BOARD_ID = "16"
GRID_SIZE = 5

# Template matching
MATCH_THRESHOLD = 0.225
MIN_DISTANCE_NMS = 25
SCALES = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
ROTATIONS = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

# Output
SAVE_DEBUG_IMAGE = True
SHOW_DEBUG_IMAGE = True
OUTPUT_DIR = BASE_DIR / "output"

# Terrains that should not count as scoring regions
IGNORE_TERRAINS = {"castle", "slott", "midt", "center", "centre", "home"}


# =========================
# CSV HELPERS
# =========================
def detect_csv_delimiter(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        sample = f.read(2048)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;").delimiter
    except Exception:
        return ";"


def load_board_scores(csv_path):
    delimiter = detect_csv_delimiter(csv_path)
    scores = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        if reader.fieldnames is None:
            raise ValueError("board_scores.csv er tom.")

        required = {"board_id", "total_score"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"board_scores.csv mangler kolonner: {missing}")

        for row in reader:
            board_id = row["board_id"].strip()
            total_score = int(row["total_score"])
            scores[board_id] = total_score

    return scores


# =========================
# KNN MODEL
# =========================
class SimpleKNN:
    def __init__(self, k=3):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int32)
        return self

    def predict_one(self, x):
        distances = np.linalg.norm(self.X - x[None, :], axis=1)
        nearest_idx = np.argsort(distances)[:self.k]
        nearest_labels = self.y[nearest_idx]
        return Counter(nearest_labels.tolist()).most_common(1)[0][0]

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.array([self.predict_one(x) for x in X], dtype=np.int32)


def load_knn_model(model_path):
    data = np.load(model_path, allow_pickle=True)

    k = int(data["k"][0])
    X = data["X"]
    y = data["y"]
    label_names = list(data["label_names"])

    model = SimpleKNN(k=k)
    model.fit(X, y)

    return model, label_names


def central_crop(img, margin_ratio=0.12):
    h, w = img.shape[:2]
    dy = int(h * margin_ratio)
    dx = int(w * margin_ratio)

    if h - 2 * dy <= 4 or w - 2 * dx <= 4:
        return img

    return img[dy:h - dy, dx:w - dx]


def extract_hsv_features(tile_bgr):
    crop = central_crop(tile_bgr)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    features = []

    for ch in range(3):
        channel = hsv[:, :, ch].astype(np.float32)
        features.append(float(channel.mean()))
        features.append(float(channel.std()))

    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [4], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [4], [0, 256]).flatten()

    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()

    features.extend(hist.tolist())
    return np.array(features, dtype=np.float32)


# =========================
# PATH HELPERS
# =========================
def get_board_path(board_id):
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        candidate = RAW_BOARD_DIR / f"{board_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def get_tile_path(board_id, row, col):
    return TILES_DIR / board_id / f"{board_id}_r{row}_c{col}.png"


# =========================
# TERRAIN PREDICTION
# =========================
def predict_terrain_grid(board_id, model, label_names, grid_size=5):
    terrain_grid = []

    for row in range(grid_size):
        terrain_row = []
        for col in range(grid_size):
            tile_path = get_tile_path(board_id, row, col)
            tile = cv2.imread(str(tile_path))

            if tile is None:
                raise FileNotFoundError(f"Fant ikke tile: {tile_path}")

            feat = extract_hsv_features(tile)
            pred_idx = model.predict([feat])[0]
            terrain_row.append(label_names[pred_idx])

        terrain_grid.append(terrain_row)

    return terrain_grid


# =========================
# TEMPLATE MATCHING
# =========================
def aggressive_nms(candidates, min_distance=25):
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    kept = []

    for candidate in candidates:
        x, y, w, h = candidate["x"], candidate["y"], candidate["w"], candidate["h"]
        cx, cy = x + w / 2.0, y + h / 2.0

        duplicate = False
        for chosen in kept:
            vx, vy, vw, vh = chosen["x"], chosen["y"], chosen["w"], chosen["h"]
            vcx, vcy = vx + vw / 2.0, vy + vh / 2.0

            distance = np.hypot(cx - vcx, cy - vcy)
            if distance < min_distance:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return kept


def detect_crowns_on_board(board_bgr, template_paths):
    board_gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    board_gray_clahe = clahe.apply(board_gray)

    board_blur = cv2.GaussianBlur(board_gray_clahe, (3, 3), 0)
    board_edges = cv2.Canny(board_blur, 30, 100)

    all_candidates = []

    for template_path in template_paths:
        if not template_path.exists():
            continue

        template_original = cv2.imread(str(template_path))
        if template_original is None:
            continue

        t_gray = cv2.cvtColor(template_original, cv2.COLOR_BGR2GRAY)

        for scale in SCALES:
            new_w = int(t_gray.shape[1] * scale)
            new_h = int(t_gray.shape[0] * scale)

            if new_w <= 0 or new_h <= 0:
                continue

            t_resized = cv2.resize(t_gray, (new_w, new_h))

            for rotation in ROTATIONS:
                temp = t_resized if rotation is None else cv2.rotate(t_resized, rotation)
                temp_edges = cv2.Canny(temp, 50, 150)
                h, w = temp_edges.shape[:2]

                if h >= board_edges.shape[0] or w >= board_edges.shape[1]:
                    continue

                result = cv2.matchTemplate(board_edges, temp_edges, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(result >= MATCH_THRESHOLD)

                for x, y in zip(xs, ys):
                    score = float(result[y, x])

                    if y + h > board_bgr.shape[0] or x + w > board_bgr.shape[1]:
                        continue

                    region_bgr = board_bgr[y:y+h, x:x+w]
                    region_hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)

                    yellow_mask = cv2.inRange(region_hsv, (26, 110, 50), (35, 255, 255))
                    yellow_ratio = cv2.countNonZero(yellow_mask) / (w * h)

                    region_edges = board_edges[y:y+h, x:x+w]
                    if region_edges.size == 0:
                        continue

                    edge_count = cv2.countNonZero(region_edges)
                    edge_density = edge_count / (w * h)

                    if (
                        score >= 0.23 and
                        0.08 < yellow_ratio < 0.35 and
                        edge_density > 0.18 and
                        h > 22
                    ):
                        all_candidates.append({
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                            "score": score
                        })

    return aggressive_nms(all_candidates, min_distance=MIN_DISTANCE_NMS)


def detections_to_crown_grid(board_bgr, detections, grid_size=5):
    h, w = board_bgr.shape[:2]
    tile_h = h / grid_size
    tile_w = w / grid_size

    crown_grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

    for det in detections:
        cx = det["x"] + det["w"] / 2.0
        cy = det["y"] + det["h"] / 2.0

        col = min(int(cx // tile_w), grid_size - 1)
        row = min(int(cy // tile_h), grid_size - 1)

        crown_grid[row][col] += 1

    return crown_grid


# =========================
# SCORING
# =========================
def compute_score(terrain_grid, crown_grid):
    rows = len(terrain_grid)
    cols = len(terrain_grid[0])

    visited = [[False] * cols for _ in range(rows)]
    total_score = 0
    breakdown = []

    for r in range(rows):
        for c in range(cols):
            terrain = str(terrain_grid[r][c]).strip().lower()

            if visited[r][c]:
                continue

            if terrain in IGNORE_TERRAINS:
                visited[r][c] = True
                continue

            queue = deque([(r, c)])
            visited[r][c] = True

            area_size = 0
            crown_sum = 0
            cells = []

            while queue:
                cr, cc = queue.popleft()
                area_size += 1
                crown_sum += int(crown_grid[cr][cc])
                cells.append((cr, cc))

                for nr, nc in [(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]:
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if visited[nr][nc]:
                            continue

                        neighbor_terrain = str(terrain_grid[nr][nc]).strip().lower()
                        if neighbor_terrain == terrain:
                            visited[nr][nc] = True
                            queue.append((nr, nc))

            region_score = area_size * crown_sum
            total_score += region_score

            breakdown.append({
                "terrain": terrain,
                "area": area_size,
                "crowns": crown_sum,
                "score": region_score,
                "cells": cells
            })

    return total_score, breakdown


# =========================
# VISUALIZATION
# =========================
def draw_detections(board_bgr, detections):
    image = board_bgr.copy()

    for det in detections:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image


def print_grid(title, grid):
    print(f"\n{title}:")
    for row in grid:
        print(row)


# =========================
# MAIN PIPELINE
# =========================
def main():
    board_path = get_board_path(BOARD_ID)
    if board_path is None:
        print(f"Fant ikke brett med board_id={BOARD_ID}")
        return

    board = cv2.imread(str(board_path))
    if board is None:
        print(f"Kunne ikke lese bildet: {board_path}")
        return

    board_scores = load_board_scores(BOARD_SCORES_PATH)
    true_score = board_scores.get(BOARD_ID)

    terrain_model, label_names = load_knn_model(MODEL_PATH)

    terrain_grid = predict_terrain_grid(BOARD_ID, terrain_model, label_names, grid_size=GRID_SIZE)

    detections = detect_crowns_on_board(board, TEMPLATE_PATHS)
    crown_grid = detections_to_crown_grid(board, detections, grid_size=GRID_SIZE)

    total_score, breakdown = compute_score(terrain_grid, crown_grid)

    print_grid("Terrain grid", terrain_grid)
    print_grid("Crown grid", crown_grid)

    print(f"\nPredikert total score: {total_score}")

    if true_score is not None:
        print(f"Fasit total score:     {true_score}")
        print(f"Avvik:                 {total_score - true_score}")
        print(f"Riktig score:          {total_score == true_score}")
    else:
        print("Fant ikke board_id i board_scores.csv")

    print("\nBreakdown:")
    for region in breakdown:
        print(
            f"- terrain={region['terrain']}, "
            f"area={region['area']}, "
            f"crowns={region['crowns']}, "
            f"score={region['score']}"
        )

    debug_image = draw_detections(board, detections)

    if SAVE_DEBUG_IMAGE:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{BOARD_ID}_detections.png"
        cv2.imwrite(str(out_path), debug_image)
        print(f"\nLagret debug-bilde til: {out_path}")

    if SHOW_DEBUG_IMAGE:
        cv2.imshow("Detections", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import os
import pandas as pd

def aggressive_nms(kandidater, min_afstand=25):
    if not kandidater:
        return []
    kandidater = sorted(kandidater, key=lambda k: k["score"], reverse=True)
    beholdte = []
    
    for kandidat in kandidater:
        x, y, w, h = kandidat["x"], kandidat["y"], kandidat["w"], kandidat["h"]
        cx, cy = x + w / 2.0, y + h / 2.0
        
        er_duplikat = False
        for valgt in beholdte:
            vx, vy, vw, vh = valgt["x"], valgt["y"], valgt["w"], valgt["h"]
            vcx, vcy = vx + vw / 2.0, vy + vh / 2.0
            
            afstand = np.hypot(cx - vcx, cy - vcy)
            if afstand < min_afstand:
                er_duplikat = True
                break
        
        if not er_duplikat:
            beholdte.append(kandidat)
    return beholdte

def detekter_alle_kroner(felt_sti, template_stier):
    felt_original = cv2.imread(felt_sti)
    if felt_original is None:
        return []

    # Konverter til gråtone (som før)
    felt_gray = cv2.cvtColor(felt_original, cv2.COLOR_BGR2GRAY)

    # Anvend CLAHE for at fremhæve detaljer
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    felt_gray_clahe = clahe.apply(felt_gray)

    # Brug nu CLAHE-billedet til Blur og Canny i stedet for det gamle felt_gray
    felt_blur = cv2.GaussianBlur(felt_gray_clahe, (3, 3), 0)
    felt_edges = cv2.Canny(felt_blur, 30, 100)
    
    alle_kandidater = []

    # 2. LOOP OVER TEMPLATES
    for template_sti in template_stier:
        template_original = cv2.imread(template_sti)
        if template_original is None: continue

        t_gray = cv2.cvtColor(template_original, cv2.COLOR_BGR2GRAY)
        
        for skala in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]:
            ny_w = int(t_gray.shape[1] * skala)
            ny_h = int(t_gray.shape[0] * skala)
            t_resized = cv2.resize(t_gray, (ny_w, ny_h))

            for vinkel in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                temp = t_resized if vinkel is None else cv2.rotate(t_resized, vinkel)
                temp_edges = cv2.Canny(temp, 50, 150)
                h, w = temp_edges.shape[:2]

                res = cv2.matchTemplate(felt_edges, temp_edges, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= 0.225) 

                for pt in zip(*loc[::-1]):
                    x, y = pt
                    score = float(res[y, x])

                    if y + h > felt_original.shape[0] or x + w > felt_original.shape[1]:
                        continue

                    # FARVE-VALIDERING
                    region_bgr = felt_original[y:y+h, x:x+w]
                    region_hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
                    # Din optimerede gul maske (Saturation 110)
                    gul_maske = cv2.inRange(region_hsv, (26, 110, 50), (35, 255, 255))
                    gul_pixel_procent = cv2.countNonZero(gul_maske) / (w * h)

                    # KANT-TÆTHED
                    region_kanter = felt_edges[y:y+h, x:x+w]
                    if region_kanter.size == 0: continue
                    kant_antal = cv2.countNonZero(region_kanter)
                    taethed = kant_antal / (w * h)
                    
                    # DIN KIRURGISKE DØRVOGTER
                    if (score >= 0.23 and 
                        0.08 < gul_pixel_procent < 0.32 and 
                        taethed > 0.18 and 
                        h > 22): 
                        
                        alle_kandidater.append({
                            "x": x, "y": y, "w": w, "h": h, 
                            "score": score
                        })

    return aggressive_nms(alle_kandidater, min_afstand=25)

def main():
    # --- KONFIGURATION (Ret disse stier!) ---
    MAPPE_STI = "/Users/gustavbrondum/Desktop/DAKI/Miniprojekt - Kingdomino/King Domino dataset/"
    CSV_STI = "/Users/gustavbrondum/Desktop/DAKI/Miniprojekt - Kingdomino/tile_annotations.csv"
    TEMPLATE_STIER = [
        "/Users/gustavbrondum/Desktop/DAKI/Miniprojekt - Kingdomino/krone_lys.jpeg",
        "/Users/gustavbrondum/Desktop/DAKI/Miniprojekt - Kingdomino/krone_mork.jpeg",
        "/Users/gustavbrondum/Desktop/DAKI/Miniprojekt - Kingdomino/krone_utydelig.jpeg",
    ]

    # --- 1. INDLÆS OG AGGREGER GROUND TRUTH ---
    print("Indlæser facitliste...")
    df = pd.read_csv(CSV_STI, sep=';')
    facit_dict = df.groupby('board_id')['crowns'].sum().to_dict()

    # --- 2. FIND BILLEDER (1.jpg til 54.jpg) ---
    billeder = [f for f in os.listdir(MAPPE_STI) if f.lower().endswith(".jpg")]
    # Numerisk sortering
    billeder.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    total_tp, total_fp, total_fn = 0, 0, 0

    print(f"\n{'Billede':<12} | {'Fundet':<8} | {'Facit':<6} | {'Status'}")
    print("-" * 45)

    for fil in billeder:
        board_id = int(''.join(filter(str.isdigit, fil)))
        if board_id not in facit_dict: continue
        if board_id > 54: continue # Vi tager kun de første 54 jf. din besked

        rigtigt_antal = facit_dict[board_id]
        sti = os.path.join(MAPPE_STI, fil)
        
        fundne_kroner = detekter_alle_kroner(sti, TEMPLATE_STIER)
        antal_fundne = len(fundne_kroner)

        tp = min(antal_fundne, rigtigt_antal)
        fp = max(0, antal_fundne - rigtigt_antal)
        fn = max(0, rigtigt_antal - antal_fundne)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        status = " OK" if antal_fundne == rigtigt_antal else f" ({antal_fundne-rigtigt_antal:+})"
        print(f"{fil:<12} | {antal_fundne:<8} | {rigtigt_antal:<6} | {status}")

    # --- 3. BEREGN OG PRINT RESULTATER ---
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("-" * 45)
    print(f"SAMLET EVALUERING (Billede 1-54):")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")
    print("-" * 45)

if __name__ == "__main__":
    main()
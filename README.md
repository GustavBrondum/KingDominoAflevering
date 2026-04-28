# KingDominoAflevering
Formålet med dette projekt er at automatisere pointberegningen i brætspillet Kingdomino ved hjælp af Computer Vision og Machine Learning. Systemet analyserer et billede af en spilleplade, identificerer terræntyper, detekterer kroner og udregner den samlede score efter spillets officielle regler.

Systemets funktioner:
Systemet er opbygget som en pipeline, der gennemfører følgende trin:
1. Terrænklassificering: Hvert tile klassificeres ved hjælp af en KNN-model baseret på HSV.
2. Detektion af kroner: Systemet finder kroner ved hjælp af multi-scale template matching. Der bruges Canny Edge Detection og farvefiltrering i et forsøg på at opretholde robusthed.
3. Region analyse: Sammenhængende felter grupperes i regioner ved hjælp af en Breadth-First Search algoritme.
4. Pointberegning: Scoren for hver region udregnes (Areal * Antal kroner) og den samlede score summereres.

Projektets struktur:
MINIPROJEKT.py: Hovedprogrammet der indeholder hele pipelinen og pointlogikken.
model/terrain_knn.npz: Den trænede KNN.model til terrængenkendelse.
Crown Templates/: Billedeskabeloner brugt til Template Matching af kroner.
board_scores.csv: Ground truth til antallet af point til hver spilleplade.
tile_annotations.csv: Ground truth til terræntype og antal kroner til alle tiles.
King domino dataset/: De rå billeder af de fulde spilleplader.
DATASETT PROCESSED/: De udklippede tiles. 

For at vælge bræt:
Åbn MINIPROJEKT.py og ret variablen BOARD_ID i toppen af koden (fx BOARD_ID = 52)

Tekniske detaljer i koden:
Billedbehandling: Vi anvender cv2.createCLAHE for at opnå bedre kontrast og cv2.Canny til kant-detektering. Dette gør vi for at forsøge at krone-detekteringen bliver mere robust over for forskellige belysningsforhold.

Feauture Extraction: Systemet udtrækker middelværdier og histogrammer i HSV-farverummet, hvilket gør det muligt for KNN-modellen at kende forskel på de forskellige terræntyper.

NMS (Non-Maximum suppression): Ved brug af funktionen aggressive_nms sikrer vi, at den samme krone ikke tælles flere gange, hvis flere skabeloner overlapper det samme område.


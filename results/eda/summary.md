# HAM10000 metadata EDA (DERM-8)

Source: `HAM10000_metadata.csv` — 10,015 images, 7,470 lesions.

| Class | Name | Images | Share | Histopathology-confirmed |
|---|---|---:|---:|---:|
| akiec | Actinic keratosis (akiec) | 327 | 3.3% | 100% |
| bcc | Basal cell carcinoma (bcc) | 514 | 5.1% | 100% |
| bkl | Benign keratosis (bkl) | 1,099 | 11.0% | 70% |
| df | Dermatofibroma (df) | 115 | 1.1% | 48% |
| mel | Melanoma (mel) | 1,113 | 11.1% | 100% |
| nv | Melanocytic nevus (nv) | 6,705 | 66.9% | 37% |
| vasc | Vascular lesion (vasc) | 142 | 1.4% | 47% |

## Key findings
- **Imbalance:** nv is 66.9% of images; df (115) and vasc (142) are the rarest. Largest/smallest class ratio = 58:1.
  A model that always predicts nv scores 66.9% accuracy, so we report balanced accuracy and per-class recall.
- **Malignant share:** akiec + bcc + mel = 19.5% of images.
- **Several images per lesion:** 1,956 of 7,470 lesions (26%) have more than one image (max 6). The split must group by `lesion_id` (DERM-10).
- **Label quality:** malignant classes are akiec 100%, bcc 100%, mel 100% biopsy-confirmed; nv only 37% (the rest by follow-up or consensus), so label noise sits on the benign side.
- **Age:** median 50 years overall; 57 missing. Median by class: akiec 70, bcc 70, bkl 65, df 50, mel 60, nv 45, vasc 55.
- **Sex:** 5,406 male, 4,552 female, 57 unknown.
- **Body site:** 15 sites; most common: back, lower extremity, trunk. 234 unknown.

Figures: class_distribution, images_per_lesion, age_by_class, female_share_by_class,
histo_confirmed_by_class, body_site (PNG, this folder).


# Extensie Experimentală: Pipeline pe Imagini Pre-Cropate cu RandomSampler3D — Comparație cu Varianta HOG-Guided

---

## Introducere

Prezenta extensie a lucrării propune și evaluează o **variantă alternativă a pipeline-ului CSNN** pentru recunoașterea acțiunilor umane pe datasetul KTH [18], în care detecția persoanei și focalizarea atenției spațiale **nu mai sunt realizate la runtime prin `HOGSampler3D`**, ci sunt **pre-calculate complet offline**, iar rețeaua spiking consumă direct **imagini deja centrate pe persoană**.

Arhitectura originală, descrisă în capitolele anterioare, se bazează pe un pipeline **video → HOG+MOG2 → JSON cu bounding box-uri → VideoKTH_3D → HOGSampler3D**, unde sampler-ul STDP primește la fiecare iterație coordonatele zonei de interes (Region of Interest — ROI) și forțează convoluția să se antreneze exclusiv pe patch-uri conținând persoana. În această variantă extinsă, **întregul pas de focalizare este mutat în etapa de pre-procesare**: cadrele sunt extrase, decupate (cropped) pe baza bounding box-ului detectat, redimensionate la rezoluția țintă și salvate ca imagini JPG. Rețeaua CSNN nu mai vede videoclipuri, ci **secvențe temporale de imagini deja curate**, iar sampler-ul intern poate reveni la varianta sa **cea mai simplă și mai biologic neutră**: `RandomSampler3D`.

Această abordare este inspirată direct din literatura clasică a SNN-urilor antrenate nesupervizat prin STDP — Masquelier & Thorpe [33], Kheradpisheh et al. [25], Mozafari et al. [26] — care operează toate pe **imagini decupate și centrate**, folosind eșantionare aleatorie pentru antrenament. Ne permite astfel să izolăm o întrebare fundamentală:

> **Este ghidarea HOG la runtime strict necesară, sau aceeași focalizare spațială poate fi obținută echivalent prin pre-procesare, eliberând sampler-ul de responsabilitatea atenției?**

Comparația dintre cele două variante — `KTH_3D.cpp` (video + HOGSampler3D) și `KTH_3D_v2.cpp` (imagini pre-cropate + RandomSampler3D) — oferă răspunsuri cuantificabile la întrebări de natură arhitecturală: unde se face cel mai bine lucrul de atenție, ce cost computațional introduce fiecare strategie, și cum variază acuratețea, timpii de execuție și complexitatea codului între cele două abordări.

**[* FIGURA E1: Diagramă side-by-side a celor două pipeline-uri. În stânga: Video brut → HOG+MOG2 → JSON bbox → VideoKTH_3D (decodează cadrele la runtime) → HOGSampler3D (aplică masca bbox) → conv1. În dreapta: Video brut → extract_frames_kth.py (detecție + crop + save JPG) → ImageSequenceKTH (citește JPG-uri) → RandomSampler3D (eșantionare uniformă) → conv1. Săgeată bidirecțională între cele două pipeline-uri cu eticheta "Comparare experimentală". *]**

---

## 1. Motivația Abordării pe Imagini Pre-Cropate

### 1.1. Limitările Pipeline-ului Video cu Sampler HOG-Guided

Pipeline-ul descris în capitolele anterioare oferă performanță competitivă (**~71%** acuratețe pe conv1 cu seed=123), dar prezintă câteva limitări structurale:

1. **Cost computațional crescut la runtime**: deși detecția HOG este pre-calculată offline, la runtime `VideoKTH_3D` trebuie să **decodeze videoclipurile complete** cu OpenCV (`cv2.VideoCapture`), să poziționeze cursorul la cadrele de interes (`cap.set(CAP_PROP_POS_FRAMES, ...)`) și să aplice redimensionarea per cadru. Decodarea H.264/MPEG-4 are un overhead semnificativ, în special pentru indexi de cadre non-secvențiali (keyframe-seeking).

2. **Cuplaj strâns între sampler și dataset**: `HOGSampler3D` depinde de sistemul de mapare statică `_train_sample_mapping` / `_test_sample_mapping` din `VideoKTH_3D`, care leagă indexul global al sample-ului curent de cheia videoclipului și grupul temporal. Această dependență cross-class complică extinderea arhitecturii la alte dataset-uri.

3. **Informație redundantă transferată prin pipeline**: cadrele originale KTH (160×120) sunt redimensionate la 80×60, din care doar **~20% din pixeli** conțin persoana. Restul sunt fundal irelevant pe care filtrul OnOff îl procesează inutil, consumând memorie și timp de execuție.

4. **Focalizare spațială aplicată ex-post**: sampler-ul HOG extrage patch-uri din zona bbox-ului, dar input-ul către filtrul OnOff și codificarea latency rămâne cadrul întreg 80×60. Filtrul spiking generează spike-uri și în zone irelevante (fundal), care sunt apoi ignorate de sampler, dar calculul este deja efectuat.

### 1.2. Avantajele Conceptuale ale Imaginilor Pre-Cropate

Mutarea integrală a focalizării spațiale în pre-procesare oferă următoarele beneficii:

1. **Input dens în informație utilă**: după crop, **~100% din pixeli** aparțin persoanei sau vecinătății sale imediate. Filtrul OnOff și codificarea latency generează spike-uri relevante peste tot cadrul, fără pierderi în zone inutile.

2. **Reducerea dimensiunii efective a scenei**: persoanele din KTH ocupă tipic 30–60% din înălțimea cadrului original. Prin crop strâns, scena este **concentrată la ~10× densitate informațională**, permițând filtrelor convoluționale să învețe trăsături mai specifice cu aceeași capacitate (96 filtre 5×5×3).

3. **Decuplarea sampler-ului de dataset**: `RandomSampler3D` nu necesită niciun context extern — eșantionează uniform aleatoriu în limitele tensorului de input. Arhitectura devine **modulară** și ușor de portat la alte dataset-uri de acțiuni umane (UCF-Sports, HMDB-51).

4. **Consistență cu literatura clasică SNN**: Masquelier & Thorpe (2007) [33], Kheradpisheh et al. (2018) [25], Mozafari et al. (2019) [26] — toate lucrările de referință în CSNN antrenat STDP operează pe **imagini decupate** cu sampling aleatoriu. Varianta propusă aliniază arhitectura cu precedentele din literatură, permițând comparații directe.

5. **Eficiență la rulare repetată**: imaginile JPG sunt mult mai rapide de decodat decât videoclipurile (sub 1 ms per cadru vs. 20–50 ms cu seeking), iar datele pot fi încărcate o singură dată în RAM pentru experimente iterative.

### 1.3. Trade-off-uri și Riscuri

Există și dezavantaje potențiale care trebuie evaluate empiric:

1. **Pierderea contextului de fundal**: unele acțiuni ar putea beneficia de informația despre scenă (ex. mediul indoor vs. outdoor pentru distincția scenarii `d1` vs. `d2` în KTH). Prin crop, această informație este eliminată.

2. **Sensibilitate crescută la calitatea detecției**: dacă bbox-ul este incorect (offset spațial, dimensiune greșită), persoana poate fi parțial sau integral exclusă din crop — spre deosebire de pipeline-ul HOG unde bbox-ul doar ghidează sampler-ul, iar persoana rămâne vizibilă în cadrul complet dacă sampler-ul cade pe o zonă greșită.

3. **Pierderea controlului de atenție adaptiv**: `HOGSampler3D` folosește un cache cu fallback temporal pe 3 niveluri, permițând adaptarea inteligentă când bbox-ul lipsește pe un cadru intermediar. Varianta cu imagini pre-cropate nu are acest mecanism — se bazează integral pe calitatea detecției offline.

---

## 2. Scriptul `extract_frames_kth.py`: Extragerea Secvențelor Consecutive

### 2.1. Obiectivul Scriptului

Scriptul Python `src/tool/extract_frames_kth.py` a fost creat de autor pentru a produce **imagini JPG cropate** din videoclipurile KTH, organizate ca **secvențe temporale consecutive** de `seq_length = 5` cadre, cu **`num_sequences = 10` secvențe non-suprapuse per videoclip**. Fiecare secvență este garantat formată din cadre **strict consecutive** în video (indici $i, i+1, i+2, i+3, i+4$), pentru a respecta condiția fundamentală a învățării STDP în 3D: filtrul temporal de adâncime `filter_depth` trebuie să capteze **mișcare reală**, nu salturi temporale.

Structura de output este următoarea:

```
kth_cropped/
├── train/
│   ├── boxing/
│   │   ├── person01_boxing_d1_frame0042.jpg   ← început secvența 1
│   │   ├── person01_boxing_d1_frame0043.jpg   ← consecutiv
│   │   ├── person01_boxing_d1_frame0044.jpg
│   │   ├── person01_boxing_d1_frame0045.jpg
│   │   ├── person01_boxing_d1_frame0046.jpg   ← sfârșit secvența 1
│   │   ├── person01_boxing_d1_frame0089.jpg   ← început secvența 2 (alt interval temporal)
│   │   └── ...
│   ├── handclapping/
│   ├── handwaving/
│   ├── jogging/
│   ├── running/
│   └── walking/
├── test/
│   └── ... (aceeași structură)
└── metadata.json                              ← metadate per cadru (bbox, frame_idx, sursă)
```

### 2.2. Arhitectura Scriptului: Reutilizarea Detectorilor HOG+MOG2

Pentru a garanta **comparabilitatea directă** cu pipeline-ul video, scriptul **reutilizează integral** algoritmii de detecție HOG + MOG2 validați în `extract_bboxes_kth.py`:

| Componentă | Sursă | Reutilizare |
|-----------|-------|-------------|
| `clip_bbox` | `extract_bboxes_kth.py:17-24` | identică |
| `passes_bbox_quality` | `extract_bboxes_kth.py:27-34` | identică |
| `bbox_iou`, `center_distance` | `extract_bboxes_kth.py:43-62` | identice |
| `normalize_confidences`, `score_box` | `extract_bboxes_kth.py:66-105` | identice |
| `merge_hog_mog2` (fuziune per-cadru) | `extract_bboxes_kth.py:107-164` | identică |
| `smooth_bbox` (EMA cu $\alpha = 0.65$) | `extract_bboxes_kth.py:166-181` | identică |
| `detect_persons_in_frame` (HOG upscalat 3×) | `extract_bboxes_kth.py:183-212` | identică |
| `detect_persons_mog2` (bg subtraction) | `extract_bboxes_kth.py:215-237` | identică |

Prin reutilizarea algoritmilor, **calitatea detecției este identică** între cele două pipeline-uri, eliminând ca variabilă confundare diferența între detectori și izolând efectul real al strategiei (runtime sampling vs. offline crop).

### 2.3. Diferența Structurală: Selecția Secvențelor Consecutive

Diferența fundamentală între `extract_bboxes_kth.py` și `extract_frames_kth.py` este **strategia de selecție a cadrelor**:

#### `extract_bboxes_kth.py` (pipeline video)

Funcția `select_centered_groups` (src/tool/extract_bboxes_kth.py:369) selectează **cele mai bune 10 grupuri temporale** după criteriul de confidență a detecției pe cadrul central, folosind offset-uri simetrice cu stride `frame_gap`:

$$\text{offsets} = [-2g,\; -g,\; 0,\; +g,\; +2g], \quad g = \texttt{frame\_gap}$$

Cu setarea default `frame_gap = 2`, un grup conține cadrele $[F_{c-4}, F_{c-2}, F_c, F_{c+2}, F_{c+4}]$ — adică **5 cadre sparse**, cu un cadru ignorat între fiecare pereche. Fereastra acoperită este de 9 cadre ($\approx 360$ ms la 25 fps), dar doar 5 sunt sampled.

#### `extract_frames_kth.py` (pipeline imagini)

Funcția `select_consecutive_sequences` (src/tool/extract_frames_kth.py:324) selectează **ferestre de 5 cadre strict consecutive** ($\text{frame\_gap} = 1$):

$$\text{offsets} = [0,\; +1,\; +2,\; +3,\; +4]$$

Algoritmul procedează în 4 pași:

1. **Identificarea cursivelor maximale** (runs): se găsesc toate secvențele contigue de cadre pentru care persoana a fost detectată cu succes (indici $i, i+1, i+2, \ldots$). Un run se termină când apare o întrerupere (cadru fără detecție), momentul la care începe un nou run.

2. **Fragmentarea în ferestre non-overlapping**: fiecare run de lungime $L$ este împărțit în $\lfloor L / \text{seq\_length} \rfloor$ ferestre non-suprapuse de exact 5 cadre consecutive.

3. **Colectarea tuturor candidaților**: toate ferestrele din toate cursivele sunt agregate într-o listă globală `all_windows`.

4. **Distribuție temporală uniformă**: dacă sunt mai multe ferestre disponibile decât `num_sequences = 10`, se aleg 10 ferestre uniform distribuite pe lungimea listei:
$$W_i = \text{all\_windows}\left[\left\lfloor i \cdot \frac{|\text{all\_windows}|}{\text{num\_sequences}} \right\rfloor\right], \quad i = 0, 1, \ldots, 9$$

Această strategie garantează că **ferestrele acoperă diferite momente ale videoclipului** (început, mijloc, sfârșit), evitând concentrarea tuturor celor 10 ferestre într-o singură zonă temporală.

### 2.4. Justificarea Consecutivității pentru STDP 3D

Motivul pentru care varianta cu imagini cere **strict consecutiv** (gap=1) în loc de stride-2 este legat de **natura învățării STDP în 3D**:

> Filtrul `conv1` are adâncime temporală $f_t = 3$, iar filtrul `conv2`/`fc1` au $f_t = 2$. STDP învață asocieri cauzale între spike-uri de intrare și spike-uri de ieșire, care se propagă **de-a lungul kernelului temporal**. Dacă cadrele sunt sparse (stride 2), filtrul nu vede mișcare continuă, ci salturi discrete — ceea ce corespunde teoretic unor acțiuni de viteză dublă, care nu sunt prezente în dataset.

La 25 fps (frame rate KTH), cadrele consecutive sunt spațiate la 40 ms unele de celelalte — spacing natural pentru mișcări umane subtile (un pas de walking ocupă ~400 ms = 10 cadre, un ciclu boxing ~300 ms = 8 cadre). Un filtru cu $f_t = 3$ acoperă 120 ms, capturând fragmente cinetice coerente.

### 2.5. Crop-ul cu Padding Aspect-Aware

O inovație a scriptului `extract_frames_kth.py` este funcția `crop_and_resize` (src/tool/extract_frames_kth.py:381), care aplică un **crop aspect-aware** înainte de redimensionarea la rezoluția țintă (80×60).

Problema clasică: bbox-urile detectate pentru persoane au aspect ratio tipic $w/h \approx 1:2$ (persoană verticală), dar rezoluția țintă are aspect ratio $80/60 = 4:3 \approx 1.33$. Un redimensionare directă ar **deforma anatomic persoana** (alungire orizontală), degradând patternurile de mișcare.

Soluția implementată:

1. **Padding inițial de bază**: bbox-ul este extins cu `padding_ratio = 0.15` (15%) în ambele direcții:
$$ew = b_w (1 + 2 \cdot 0.15) = 1.3 \cdot b_w$$

2. **Expansiune aspect-aware**: se verifică dacă aspect-ul curent al crop-ului se potrivește cu cel al rezoluției țintă:
$$\text{aspect}_{current} = \frac{ew}{eh}, \quad \text{aspect}_{target} = \frac{80}{60} = 1.33$$

- Dacă $\text{aspect}_{current} < \text{aspect}_{target}$ (crop prea îngust), se expandează **orizontal**:
$$ew_{new} = eh \cdot \text{aspect}_{target}$$
- Dacă $\text{aspect}_{current} > \text{aspect}_{target}$ (crop prea lat), se expandează **vertical**:
$$eh_{new} = \frac{ew}{\text{aspect}_{target}}$$

3. **Clipping la limitele cadrului**: coordonatele extinse sunt limitate la $[0, W] \times [0, H]$.

4. **Redimensionare finală**: `cv2.resize` cu interpolare `INTER_AREA` la 80×60 pixeli.

Rezultatul este o imagine **fără deformări anatomice**, cu persoana centrată și padding uniform de context. Dacă bbox-ul iese parțial din cadru (persoană la margine), clipping-ul garantează că output-ul rămâne valid, chiar dacă padding-ul este asimetric.

**[* FIGURA E2: Diagramă ilustrativă a procesului crop_and_resize: (a) cadrul original 80×60 cu bbox raw (dreptunghi roșu), (b) bbox cu padding 15% (dreptunghi galben), (c) expansiune aspect-aware pentru a ajunge la 4:3 (dreptunghi verde), (d) crop-ul final după clipping (imagine separată), (e) rezultatul resized la 80×60. *]**

### 2.6. Carry Temporal pentru Detecții Missing

Ca și în `extract_bboxes_kth.py`, scriptul menține un mecanism de **carry temporal**: dacă detecția eșuează pe un cadru intermediar, se reutilizează ultimul bbox valid timp de până la 2 cadre consecutive (`miss_streak < 2`). Aceasta asigură continuitatea secvențelor extrase, prevenind ruperea unui run cinetic din cauza unei detecții ratate pe 1-2 cadre.

### 2.7. Filename-uri Zero-Padded pentru Sortare Determinstă

Un detaliu crucial pentru corectitudinea ulterioară în `ImageSequenceKTH`: fișierele JPG sunt salvate cu **index zero-padded la 4 cifre**:

```
person01_boxing_d1_frame0042.jpg
person01_boxing_d1_frame0043.jpg
...
person01_boxing_d1_frame0999.jpg
```

Această convenție garantează că **sortarea lexicografică** (`std::sort` pe vectorul de path-uri C++) produce **exact ordonarea temporală**, fără a fi nevoie de parsare numerică explicită. Un cadru cu index 42 ar fi sortat după un cadru cu index 421 în sortare lex fără padding, ceea ce ar rupe consecutivitatea.

### 2.8. Output JSON de Metadate

Suplimentar față de imaginile JPG, scriptul salvează `kth_cropped/metadata.json` cu informații complete per cadru:

```json
{
  "config": {
    "frame_width": 80, "frame_height": 60,
    "crop_width": 80, "crop_height": 60,
    "seq_length": 5, "num_sequences": 10,
    "padding_ratio": 0.15
  },
  "videos": {
    "train/boxing/person01_boxing_d1_uncomp.avi": {
      "person_id": 1, "action": "boxing", "scenario": "d1",
      "num_frames_extracted": 50, "num_sequences": 10,
      "frames": [
        {
          "filename": "person01_boxing_d1_frame0042.jpg",
          "frame_idx": 42,
          "bbox": {"x": 15, "y": 8, "w": 30, "h": 45, "confidence": 0.82, "source": "hog+mog2"}
        },
        ...
      ]
    }
  }
}
```

Acest fișier permite debugging, reconstituirea originii fiecărui crop și analiza post-hoc a calității detecției.

### 2.9. Comanda de Rulare

```bash
python3 src/tool/extract_frames_kth.py \
    --input_path /home/mmuntean/kth_organized/ \
    --output_dir /home/mmuntean/kth_cropped/ \
    --seq_length 5 \
    --num_sequences 10 \
    --frame_width 80 --frame_height 60 \
    --crop_width 80 --crop_height 60 \
    --padding_ratio 0.15 \
    --hit_threshold -0.75 \
    --mog2_min_area 180 \
    --min_bbox_area_ratio 0.008 \
    --min_bbox_aspect 0.22 \
    --max_bbox_aspect 1.6
```

**[* FIGURA E3: Strip orizontal cu 5 cadre consecutive cropate pentru fiecare din cele 6 acțiuni KTH (boxing, handclapping, handwaving, jogging, running, walking). Fiecare rând = o acțiune, cu o etichetă și numele videoclipului sursă. Arată că persoana este centrată și proporțional similară în toate cropurile. *]**

---

## 3. Clasa C++ `ImageSequenceKTH`: Input Stream pentru Imagini Pre-Cropate

### 3.1. Obiectivul Clasei

Clasa `ImageSequenceKTH` (`include/dataset/ImageSequenceKTH.h`, `src/dataset/ImageSequenceKTH.cpp`) a fost creată de autor ca **echivalent direct al `VideoKTH_3D`**, dar care citește **imagini JPG pre-cropate** în loc de cadre decodate din videoclipuri. Semantic, cele două clase sunt interschimbabile ca input pentru același pipeline CSNN, diferența fiind doar sursa datelor.

### 3.2. Structuri Interne

```cpp
struct Sample {
    std::string action;                   // eticheta: "boxing", "running", etc.
    std::vector<std::string> frame_paths; // temporal_depth path-uri (5 JPG-uri)
};
```

Fiecare `Sample` reprezintă **o secvență temporală completă** — 5 cadre consecutive decupate dintr-un videoclip. Spre deosebire de `VideoKTH_3D` care menține structuri ierarhice complexe (`VideoHOGData → VideoGroup → FrameBBox`) și mapări statice pentru sampler, `ImageSequenceKTH` are o structură **plată**: un simplu vector de sample-uri.

### 3.3. Construcția Sample-urilor: `build_samples`

Metoda `build_samples` (src/dataset/ImageSequenceKTH.cpp:68) scanează directorul de input (train/ sau test/) și construiește lista de sample-uri în trei pași:

#### Pasul 1: Iterarea pe Subdirectoare (Acțiuni)

```cpp
for (const auto &action_entry : std::filesystem::directory_iterator(_folder_path))
{
    std::string action = action_entry.path().filename().string();  // "boxing", etc.
    ...
}
```

#### Pasul 2: Gruparea Fișierelor pe Videoclipul Sursă

Funcția auxiliară `video_key_from_filename` (src/dataset/ImageSequenceKTH.cpp:14) extrage **cheia video** dintr-un nume de fișier:

$$\text{"person01\_boxing\_d1\_frame0042.jpg"} \xrightarrow{\text{split}} \text{"person01\_boxing\_d1"}$$

Algoritmul: strip extensia `.jpg`, apoi strip sufixul `_frame0042`. Toate cadrele cu aceeași cheie video sunt agregate într-un `std::map<std::string, std::vector<std::string>>`:

```cpp
std::map<std::string, std::vector<std::string>> video_frames;
for (const auto &file_entry : std::filesystem::directory_iterator(action_entry.path()))
{
    std::string fname = file_entry.path().filename().string();
    std::string vkey = video_key_from_filename(fname);
    video_frames[vkey].push_back(file_entry.path().string());
}
```

#### Pasul 3: Fragmentarea pe Chunks de `temporal_depth`

Pentru fiecare videoclip identificat, se sortează lexicografic path-urile (= sortare temporală datorită zero-padding-ului), apoi se împart în **chunks non-suprapuse de exact `temporal_depth = 5` cadre**:

```cpp
std::sort(paths.begin(), paths.end());
size_t num_full_groups = paths.size() / _temporal_depth;
for (size_t g = 0; g < num_full_groups; ++g)
{
    Sample s;
    s.action = action;
    for (size_t f = 0; f < _temporal_depth; ++f)
        s.frame_paths.push_back(paths[g * _temporal_depth + f]);
    _samples.push_back(std::move(s));
}
```

Deoarece scriptul Python salvează exact secvențe de 5 cadre consecutive, fiecare chunk de 5 path-uri corespunde **exact la o fereastră temporală cinetic coerentă**. Nu există "coincidențe" de temporal splitting — toate chunk-urile sunt secvențe valide.

#### Pasul 4: Sortare Globală Deterministă

La final, toate sample-urile sunt sortate după (acțiune, primul path), pentru a garanta **ordonare deterministă** între rulări diferite:

```cpp
std::sort(_samples.begin(), _samples.end(), [](const Sample &a, const Sample &b) {
    if (a.action != b.action) return a.action < b.action;
    return a.frame_paths[0] < b.frame_paths[0];
});
```

### 3.4. Interfața `Input`: `next()` și `has_next()`

Clasa implementează interfața abstractă `Input` a simulatorului:

```cpp
bool has_next() const override {
    return _cursor < std::min(_samples.size(), _max_read);
}

std::pair<std::string, Tensor<InputType>> next() override {
    const Sample &sample = _samples[_cursor];
    std::pair<std::string, Tensor<InputType>> out(sample.action, _shape);
    for (size_t t = 0; t < sample.frame_paths.size(); ++t) {
        cv::Mat frame = cv::imread(sample.frame_paths[t], cv::IMREAD_COLOR);
        if (_frame_size_width != 0 && _frame_size_height != 0)
            cv::resize(frame, frame, cv::Size(_frame_size_width, _frame_size_height));
        if (_grey == 1)
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        // ... populate tensor ...
    }
    _cursor++;
    return out;
}
```

Output-ul `next()` este un tensor de formă $[H \times W \times C \times T] = [60 \times 80 \times 1 \times 5]$, identic ca shape cu cel produs de `VideoKTH_3D`. Astfel, pipeline-ul downstream (filtru OnOff, latency coding, conv1, etc.) funcționează **fără modificare** pentru orice input source.

### 3.5. Simplitatea Comparativ cu `VideoKTH_3D`

Tabelul următor ilustrează reducerea de complexitate obținută prin mutarea detecției offline:

| Aspect | `VideoKTH_3D` | `ImageSequenceKTH` |
|--------|---------------|---------------------|
| Parsare input | Parser JSON manual + OpenCV VideoCapture | `std::filesystem` + `cv::imread` |
| Structuri de date | 3 niveluri: `VideoHOGData → VideoGroup → FrameBBox` | 1 nivel: `Sample` cu vector de path-uri |
| Mapări statice | `_train_sample_mapping`, `_test_sample_mapping`, `_hog_data` | niciuna |
| Dependențe runtime | JSON + toate videoclipurile `.avi` decodate | doar directorul cu JPG-uri |
| Linii de cod | ~450 (`.cpp` + `.h`) | ~210 (`.cpp` + `.h`) |
| Overhead per `next()` | decodare video + seek + resize | 5 × `imread` (cached în page cache OS) |

Reducerea la jumătate a codului și eliminarea mapărilor statice cross-class simplifică semnificativ întreținerea și extensibilitatea.

---

## 4. Experimentul `KTH_3D_v2.cpp`: Configurația CSNN pe Imagini

### 4.1. Diferențele față de `KTH_3D.cpp`

Fișierul `apps/kth/KTH_3D_v2.cpp` a fost creat ca **variantă paralelă** a experimentului original (`apps/kth/KTH_3D.cpp`), păstrând **toată configurația CSNN identică**, dar înlocuind:

1. **Input-ul**: `VideoKTH_3D` → `ImageSequenceKTH`
2. **Sampler-ul**: `HOGSampler3D` → `RandomSampler3D` pe toate cele trei straturi (`conv1`, `conv2`, `fc1`)

Toate celelalte componente (filtru OnOff, MaxScaling, LatencyCoding, parametri STDP, threshold-uri, annealing, t_obj, output pooling, analize SVM) rămân **identice**, pentru a izola efectul real al strategiei de sampling.

### 4.2. Parametrii Experimentali

```cpp
int seed = 123;                       // determinist, identic cu KTH_3D.cpp
size_t frame_size_width = 80;
size_t frame_size_height = 60;
size_t video_frames = 5;              // = temporal_depth = seq_length
size_t tmp_filter_size = 2;           // adâncime temporală conv2/fc1
size_t temp_stride = 1;

experiment.push<process::DefaultOnOffFilter>(7, 1.0, 4.0);  // identic
experiment.push<process::MaxScaling>();                      // identic
experiment.push<LatencyCoding>();                            // identic
```

### 4.3. Stratul `conv1`: 96 Filtre, Kernel 5×5×3, RandomSampler3D

```cpp
auto &conv1 = experiment.push<layer::ConvolutionSampler3D>(96, 5, 5, 3, "", 1, 1, 1);
conv1.parameter<uint32_t>("epoch").set(150);
conv1.parameter<float>("annealing").set(0.95f);
conv1.parameter<float>("t_obj").set(0.75f);
conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(8.0, 0.1);
conv1.parameter<STDP>("stdp").set<stdp::Biological>(0.1f, 0.1f);
conv1.parameter<Sampler>("sampler").set<sampler::RandomSampler3D>();
```

Numărul de filtre (96), dimensiunea kernelului spațial (5×5), adâncimea temporală (3), numărul de epoci (150), annealing-ul (0.95) și pragul inițial Gaussian(8.0, 0.1) sunt **identice** cu `KTH_3D.cpp`. Singura diferență este sampler-ul.

### 4.4. Straturile `conv2` și `fc1`: Parametri Baseline

Urmând configurația validată experimental în varianta originală (care a obținut ~71% acuratețe pe conv1):

| Strat | Filtre | Kernel | Pool output | Threshold (Gaussian) | Epochs | t_obj |
|-------|--------|--------|-------------|---------------------|--------|-------|
| conv1 | 96 | 5×5×3 | SumPooling(2,2) | (8.0, 0.1) | 150 | 0.75 |
| pool1 | — | 2×2×1 stride 2 | — | — | — | — |
| conv2 | 64 | 5×5×2 | SumPooling(2,2) | (10.0, 0.5) | 100 | 0.60 |
| pool2 | — | 2×2×1 stride 2 | — | — | — | — |
| fc1 | 256 | 12×17×2 | (fără pooling) | (15.0, 0.1) | 100 | 0.75 |

### 4.5. Ieșirile și Analizele

```cpp
auto &conv1_out = experiment.output<TimeObjectiveOutput>(conv1, 0.75f);
conv1_out.add_postprocessing<process::SumPooling>(2, 2);
conv1_out.add_postprocessing<process::FeatureScaling>();
conv1_out.add_analysis<analysis::Activity>();
conv1_out.add_analysis<analysis::Coherence>();
conv1_out.add_analysis<analysis::Svm>();

auto &conv2_out = experiment.output<TimeObjectiveOutput>(conv2, 0.60f);
conv2_out.add_postprocessing<process::SumPooling>(2, 2);
// ... identic ...

auto &fc1_out = experiment.output<TimeObjectiveOutput>(fc1, 0.75f);
fc1_out.add_postprocessing<process::FeatureScaling>();
fc1_out.add_analysis<analysis::Activity>();
fc1_out.template add_analysis<analysis::SvmQualitative>();
```

Analiza cantitativă este efectuată la toate cele trei ieșiri prin `Svm`, iar la stratul final `fc1` este adăugată suplimentar `SvmQualitative` pentru evaluarea per-sample și generarea vizualizărilor feature map în punctele cu clasificare corectă / incorectă.

---

## 5. Protocolul Comparativ Experimental

### 5.1. Variabila Controlată

Pentru a izola efectul real al strategiei de sampling, toți ceilalți factori sunt menținuți constanți între cele două experimente:

| Factor | KTH_3D (video) | KTH_3D_v2 (imagini) |
|--------|----------------|---------------------|
| Seed | 123 | 123 |
| Rezoluție input | 80×60 | 80×60 |
| Temporal depth input | 5 | 5 |
| Kernel conv1 | 5×5×3 | 5×5×3 |
| Filtre conv1 | 96 | 96 |
| Threshold conv1 | Gaussian(8.0, 0.1) | Gaussian(8.0, 0.1) |
| Epochs conv1 | 150 | 150 |
| Annealing conv1 | 0.95 | 0.95 |
| t_obj conv1 | 0.75 | 0.75 |
| STDP tip | Biological(0.1, 0.1) | Biological(0.1, 0.1) |
| SumPooling output | (2, 2) | (2, 2) |
| **Detector HOG+MOG2** | **identic** | **identic** |
| **Bbox per cadru** | **identic** | **identic** |
| **Sampler** | **HOGSampler3D (runtime)** | **RandomSampler3D (uniform)** |
| **Locul crop-ului** | **— (fără crop)** | **offline** |

Singurele variabile experimentale sunt **sampler-ul** și **locul aplicării crop-ului**. Orice diferență de performanță reflectă exclusiv acest contrast.

### 5.2. Metrici de Evaluare

Pentru fiecare experiment se vor măsura:

1. **Acuratețe SVM per strat** (conv1, conv2, fc1) — metricul principal de discriminabilitate
2. **Timp total de execuție** (ore) — de la `./KTH_3D` până la output SVM final
3. **Timp per epocă conv1** (secunde) — să izolăm overhead-ul de I/O
4. **Timp pre-procesare offline** (minute) — `extract_frames_kth.py` vs. `extract_bboxes_kth.py`
5. **Consumul maxim de RAM** (GB) — pentru a confirma că ambele variante rulează pe hardware comun
6. **Sparsitate output conv1** — fracțiunea de valori nenule în feature map
7. **Coherence Q3 și Max** — diversitatea filtrelor învățate
8. **Matrice de confuzie per clasă** — identificarea confuziilor sistematice

### 5.3. Tabelul Rezultatelor (pentru completare de către autor)

#### 5.3.1. Acuratețe SVM

| Strat | KTH_3D (video + HOG) | KTH_3D_v2 (imagini + Random) | Diferență |
|-------|----------------------|------------------------------|-----------|
| conv1 | `[* TBD *]` % | `[* TBD *]` % | `[* TBD *]` |
| conv2 | `[* TBD *]` % | `[* TBD *]` % | `[* TBD *]` |
| fc1 | `[* TBD *]` % | `[* TBD *]` % | `[* TBD *]` |

#### 5.3.2. Timpi de Execuție

| Etapă | KTH_3D | KTH_3D_v2 |
|-------|--------|-----------|
| Pre-procesare offline | `[* TBD *]` min | `[* TBD *]` min |
| Dimensiune artefact offline | `[* TBD *]` MB (JSON) | `[* TBD *]` MB (JPG-uri) |
| Încărcare dataset în RAM | `[* TBD *]` s | `[* TBD *]` s |
| Antrenare conv1 (150 epoci) | `[* TBD *]` h | `[* TBD *]` h |
| Antrenare conv2 (100 epoci) | `[* TBD *]` h | `[* TBD *]` h |
| Antrenare fc1 (100 epoci) | `[* TBD *]` h | `[* TBD *]` h |
| Test + SVM final | `[* TBD *]` min | `[* TBD *]` min |
| **Total end-to-end** | `[* TBD *]` h | `[* TBD *]` h |

#### 5.3.3. Consum Memorie și Metrici Activitate

| Metrică | KTH_3D | KTH_3D_v2 |
|---------|--------|-----------|
| Peak RAM (GB) | `[* TBD *]` | `[* TBD *]` |
| Sparsity conv1 | `[* TBD *]` | `[* TBD *]` |
| Sparsity conv2 | `[* TBD *]` | `[* TBD *]` |
| Active units conv1 (%) | `[* TBD *]` | `[* TBD *]` |
| Coherence Q3 conv1 | `[* TBD *]` | `[* TBD *]` |
| Coherence Max conv1 | `[* TBD *]` | `[* TBD *]` |
| Mean weights conv1 | `[* TBD *]` | `[* TBD *]` |

#### 5.3.4. Acuratețe Per Clasă la `fc1`

| Clasă | KTH_3D (%) | KTH_3D_v2 (%) |
|-------|-----------|----------------|
| boxing | `[* TBD *]` | `[* TBD *]` |
| handclapping | `[* TBD *]` | `[* TBD *]` |
| handwaving | `[* TBD *]` | `[* TBD *]` |
| jogging | `[* TBD *]` | `[* TBD *]` |
| running | `[* TBD *]` | `[* TBD *]` |
| walking | `[* TBD *]` | `[* TBD *]` |

**[* FIGURA E4: Bar chart side-by-side cu acuratețea per clasă pentru cele două variante. Fiecare acțiune are două bare (verde = KTH_3D, albastru = KTH_3D_v2), valorile etichetate deasupra barelor. *]**

**[* FIGURA E5: Două matrici de confuzie plasate side-by-side: stânga = KTH_3D (video + HOG), dreapta = KTH_3D_v2 (imagini + Random). Normalizate pe rând pentru vizualizarea procentelor. *]**

### 5.4. Ipotezele de Testat

Pe baza analizei teoretice din Secțiunea 1, propunem 4 ipoteze care vor fi confirmate sau infirmate de rezultate:

**Ipoteza H1 (acuratețe apropiată)**: $|\text{Acc}_{\text{v2}} - \text{Acc}_{\text{v1}}| < 3\%$ pe conv1.

> *Justificare*: Ambele pipeline-uri aplică aceeași focalizare pe persoană; diferența este doar locul unde se face (runtime vs. offline). Dacă algoritmii de detecție sunt identici, rețeaua ar trebui să învețe aceleași trăsături.

**Ipoteza H2 (timp total redus cu 20–40%)**: $T_{\text{v2}} < 0.80 \cdot T_{\text{v1}}$.

> *Justificare*: Eliminarea decodării video și a seeking-ului non-secvențial la runtime reduce substantial overhead-ul de I/O.

**Ipoteza H3 (sparsitate crescută pentru v2)**: $\text{Sparsity}_{\text{v2}} > \text{Sparsity}_{\text{v1}}$ la conv1.

> *Justificare*: Imaginile cropate au densitate informațională mai mare, iar filtrul OnOff generează mai mulți spike-uri relevante, ducând la activări mai distribuite.

**Ipoteza H4 (coherence similară)**: diversitatea filtrelor învățate este comparabilă între cele două variante.

> *Justificare*: Numărul de filtre (96), dimensiunea kernelului (5×5×3) și regula STDP sunt identice, deci spațiul de soluții este același.

---

## 6. Implicații și Discuție

### 6.1. Decuplarea Responsabilităților

O consecință arhitecturală importantă a variantei `KTH_3D_v2.cpp` este **decuplarea netă între cele trei preocupări**:

1. **Detecția persoanei** (cine / unde) — responsabilitate exclusivă a `extract_frames_kth.py`
2. **Reprezentarea de input** (cum e structurat tensorul) — responsabilitate a `ImageSequenceKTH`
3. **Învățarea STDP** (ce trăsături emerg) — responsabilitate a `conv1/conv2/fc1` + `RandomSampler3D`

În varianta video, aceste trei preocupări erau parțial împletite: `HOGSampler3D` trebuia să cunoască structura dataset-ului (prin mapări statice), iar `VideoKTH_3D` trebuia să cunoască semantica grupurilor temporale. În noua variantă, fiecare componentă are un contract curat și poate fi testată sau înlocuită independent.

### 6.2. Aplicabilitate la Alte Dataset-uri

Arhitectura `ImageSequenceKTH + RandomSampler3D` este **complet agnostic la dataset**: orice set de acțiuni poate fi pregătit în formatul `{action}/{video_key}_frame{idx:04d}.jpg`, iar același cod CSNN va funcționa. Dataset-uri candidate pentru evaluare viitoare:

- **Weizmann Actions** [34]: 90 videouri, 9 acțiuni, similar cu KTH dar cu condiții mai variate
- **UCF-Sports** [35]: 150 videouri, 10 acțiuni sportive
- **HMDB-51** [36]: 6,849 videouri, 51 acțiuni — benchmark mai dificil

### 6.3. Limitări Care Rămân

Varianta `KTH_3D_v2.cpp` nu rezolvă toate limitările CSNN-urilor nesupervizate discutate în capitolul 12:

1. Absența învățării ierarhice adaptive rămâne (STDP este independent per strat)
2. Sensibilitatea la calitatea bbox-urilor detectate este neschimbată (doar mutată temporal)
3. Cost computațional al STDP rămâne dominat de numărul de sample-uri × epoci, nu de I/O

Totuși, simplificarea arhitecturală și alinierea cu literatura deschid calea pentru experimente viitoare:
- Extinderea la imagini RGB (color OnOff filter) fără a modifica dataset-ul
- Augmentare spațială (flip orizontal, mici traslații) aplicabilă direct peste JPG-uri
- Combinare cu R-STDP [26] pentru semi-supervizare pe imaginile cropate

---

## 7. Concluzii Experimentale

### 7.1. Contribuția Științifică

Extensia propusă oferă o contribuție metodologică prin **compararea sistematică** a două strategii fundamentale de focalizare spațială în CSNN-uri:

- **Strategia 1 — "atenție runtime"**: sampler-ul modifică dinamic zona de interes pe care se aplică STDP (HOGSampler3D + VideoKTH_3D)
- **Strategia 2 — "atenție offline"**: input-ul este pre-filtrat pentru a conține doar zona de interes (extract_frames_kth + ImageSequenceKTH + RandomSampler3D)

Cele două strategii sunt **matematic echivalente în limită** (ambele ar trebui să producă aceiași spike-uri STDP pe aceleași zone spațiale), dar diferă semnificativ în:

- Complexitate de cod (210 vs. 450 LoC pentru input class)
- Overhead computațional la runtime (I/O video vs. imread JPG)
- Modularitate și portabilitate la alte dataset-uri
- Aliniere cu literatura standard SNN

### 7.2. Recomandare Metodologică

Dacă acuratețile se dovedesc a fi comparabile (Ipoteza H1 confirmată), **varianta `KTH_3D_v2.cpp` este recomandată ca baseline pentru lucrările viitoare** din următoarele motive:

1. Cod mai simplu, mai ușor de întreținut
2. Compatibilitate cu literatura dominantă (Masquelier, Kheradpisheh, Mozafari)
3. Portabilitate la dataset-uri noi prin simpla modificare a scriptului de pre-procesare
4. Artefact offline inspectabil (JPG-urile pot fi vizualizate și verificate manual)

Dacă, dimpotrivă, `HOGSampler3D` obține consistent acuratețe semnificativ mai bună (diferență > 5%), aceasta ar demonstra **valoarea atenției adaptive la runtime**, sugerând că sampler-ul folosește informație contextuală de run-time pe care crop-ul offline nu o poate capta (ex. adaptarea la bbox-uri missing prin cache multi-nivel).

**[* FIGURA E6: Graph cu evoluția acurateței SVM pe conv1 în funcție de epoca de antrenare, două linii (KTH_3D verde, KTH_3D_v2 albastru). Axa X: epocă (0–150), axa Y: acuratețe SVM (%). Marchează punctul de convergență pentru fiecare variantă. *]**

**[* FIGURA E7: Vizualizare comparativă a filtrelor învățate (weights) pe conv1 pentru cele 2 variante. Două grilaje 8×12 (= 96 filtre), stânga KTH_3D, dreapta KTH_3D_v2. Fiecare filtru afișat ca diferența ON-OFF pe cele 3 momente temporale. Comentează similitudinile și diferențele vizuale. *]**

**[* FIGURA E8: Feature maps de la conv1 pe aceleași 6 sample-uri reprezentative (câte unul per acțiune), procesate de ambele variante. Permite vizualizarea directă a modului în care fiecare pipeline "vede" aceeași scenă. *]**

---

## 8. Anexă: Structura Fișierelor Adăugate

| Fișier | Rol | Linii |
|--------|-----|-------|
| `src/tool/extract_frames_kth.py` | Script Python: detectează persoana, extrage secvențe consecutive, crop, salvare JPG | ~670 |
| `include/dataset/ImageSequenceKTH.h` | Header C++: clasă de input pentru secvențe de imagini | ~80 |
| `src/dataset/ImageSequenceKTH.cpp` | Implementare C++: parser filesystem, sortare, tensor filling | ~210 |
| `apps/kth/KTH_3D_v2.cpp` | Aplicație C++: experimentul CSNN pe imagini pre-cropate | ~180 |

**Total: ~1,140 linii de cod adăugate**, majoritar documentate inline și cu separare clară a responsabilităților.

---

## Bibliografie Specifică Extensiei

[33] T. Masquelier, S. Thorpe, "Unsupervised learning of visual features through spike timing dependent plasticity", *PLOS Computational Biology*, 2007.

[34] M. Blank, L. Gorelick, E. Shechtman, M. Irani, R. Basri, "Actions as Space-Time Shapes", *ICCV*, 2005.

[35] M. Rodriguez, J. Ahmed, M. Shah, "Action MACH: A Spatio-temporal Maximum Average Correlation Height Filter for Action Recognition", *CVPR*, 2008.

[36] H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, T. Serre, "HMDB: A large video database for human motion recognition", *ICCV*, 2011.

---

*Acest document extinde lucrarea principală (`KTH_3D_Pipeline_Licenta.md`) cu o variantă experimentală orientată pe imagini pre-cropate. Toate metricile marcate `[* TBD *]` urmează să fie completate de autor după rularea efectivă a celor două experimente pe hardware-ul target (Google Cloud VM, 8 vCPU, 32 GB RAM).*
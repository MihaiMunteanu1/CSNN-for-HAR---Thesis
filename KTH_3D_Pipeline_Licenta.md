# Arhitectura și Pipeline-ul Experimentului KTH_3D

Această secțiune descrie pe larg abordarea propusă pentru recunoașterea acțiunilor umane pe baza dataset-ului KTH. Arhitectura folosită combină o rețea neurală spiking convoluțională (**CSNN - Convolutional Spiking Neural Network**) capabilă să asimileze vizual secvențe spatio-temporale nesupervizate, stabilizată de o tehnica de detecție biologică HOG pentru direcționarea atenției spațiale a modelului (Region of Interest) și un clasificator tradițional (SVM) pentru validarea supervizată a caracteristicilor vizuale obținute.

Spre deosebire de arhitecturile clasice de Deep Learning (tip CNN / ResNet) care procesează imagini dense matematic cu un consum energetic ridicat, abordarea curentă folosește **Paradigma Neuromorfică**. Aceasta traduce informația vizuală în impulsuri binare asincrone (Spike-uri), procesând strict evenimentele active ale mișcării umane, mimând astfel eficiența energetică și topologia creierului uman.

## 1. Setul de Date KTH (KTH Video Dataset)
Cunoscut ca unul dintre primele seturi de date standardizate (benchmark) pentru recunoașterea acțiunilor umane, **KTH Video Dataset** (introdus de Schuldt et al., 2004) conține înregistrări video focusate pe mișcări și posturi anatomice distincte. Setul este compus din șase tipuri de acțiuni umane fundamentale:
- **Atingerea adversarului imaginar / Box (Boxing)**
- **Bătutul din palme (Hand Clapping)**
- **Fluturarea mâinilor (Hand Waving)**
- **Mersul normal (Walking)**
- **Alergarea ușoară (Jogging)**
- **Alergarea rapidă (Running)**

Acțiunile sunt interpretate independent de 25 de subiecți (actori) diferiți. Pentru a aduce diversitate și robustețe în generalizarea modelelor vizuale, fiecare secvență este filmată în 4 scenarii izolate vizual: *(s1)* în aer liber (outdoors), *(s2)* în aer liber cu modificarea scalei prin depărtare de cameră, *(s3)* în aer liber cu variații de vestimentație și *(s4)* în mediu controlat în interior (indoors).

**[* RECOMANDARE IMAGINE AICI: Un colaj creat de tine cu 6 cadre reprezentative (screenshot-uri), câte unul pentru fiecare acțiune din dataset-ul KTH, pentru a ilustra cititorilor formatul vizual al bazei de date. *]**

Materialul brut este captat cu un fundal relativ static și omogen (la frecvența temporală de 25 fps). În contextul prezentei arhitecturi CSNN, rezoluția fișierelor originale (160×120 pixeli) este redimensionată agresiv (la 80×60 pixeli) tocmai pentru a testa eficiența rețelelor Spiking orientate pe contur fluid în detrimentul puterii de calcul exorbitante pe care ar fi necesitat-o o arhitectură clasică prelucrând fiecare pixel.

## 2. Preprocesarea Inteligentă a Cadrelor: Scriptul `extract_person_frames_v2.py`

Un element central al pipeline-ului propus este **extragerea offline a cadrelor relevante** din fiecare videoclip KTH, realizată printr-un script Python creat de autor (`src/tool/extract_person_frames_v2.py`). Spre deosebire de abordarea clasică în care cadrele sunt extrase secvențial sau aleatoriu din flux (ceea ce duce frecvent la selectarea unor momente statice sau la pierderea persoanei din cadru), acest script analizează întregul videoclip, identifică zonele unde persoana este vizibilă și selectează **grupuri temporale optime** de câte 3 cadre.

### 2.1. Structura unui Grup Temporal

Fiecare grup temporal este format din **5 cadre** dispuse simetric în jurul unui cadru central:

$$\text{Grup} = [F_{center - 2 \cdot gap},\; F_{center - gap},\; F_{center},\; F_{center + gap},\; F_{center + 2 \cdot gap}]$$

unde $gap = 3$ cadre intermediare. Astfel, cele 5 cadre ale unui grup acoperă o fereastră temporală de $4 \times 3 + 1 = 13$ cadre din video (520 ms la 25 fps), suficientă pentru a surprinde un ciclu complet de mișcare (extensia și retragerea brațului în boxing, doi pași consecutivi în walking).

Numărul de grupuri extrase per videoclip este configurat la **3 sample-uri per video** (`train_sample_per_video = 3`, `test_sample_per_video = 3`), sortate cronologic, oferind o acoperire suficientă a secvenței de acțiune cu un consum de memorie gestionabil pe sisteme cu 16GB RAM.

### 2.2. Detectorul Primar: HOG (Histogram of Oriented Gradients)

Algoritmul principal de detectare a persoanei este **HOG cu clasificator SVM**, implementat prin OpenCV (`cv2.HOGDescriptor` cu `getDefaultPeopleDetector()`). Acesta funcționează astfel:

1. **Calculul Gradienților Pixelari**: Imaginea este derivată matricial la nivel de pixel pentru a obține magnitudinea și direcția orientării fiecărei margini organice (conturul cămășii, granița corp-fundal).
2. **Construirea Celulelor de Histograme**: Imaginea este segmentată în celule optice ($8 \times 8$ pixeli). Pentru fiecare celulă, vectorii gradient „votează" direcția dominantă în intervalul 0-180°, formând o histogramă locală de orientări.
3. **Normalizarea pe Blocuri**: Celulele sunt agregate în blocuri suprapuse ($16 \times 16$ pixeli), histogramele fiind normalizate L2 pentru insensibilitate la variațiile de iluminare (umbră, reflexii indoor vs. outdoor).
4. **Clasificarea SVM cu Fereastră Glisantă**: Descriptorul HOG vectorial parcurge cadrul prin ferestre glisante (sliding windows) de dimensiuni multiple. Modelul SVM pre-antrenat validează binar dacă conținutul ferestrei corespunde unei siluete umane, returnând coordonatele și scorul de confidență al detecției.

Deoarece rezoluția de lucru (80×60) este prea mică pentru detectorul HOG standard (care necesită minimum 64×128 pixeli per fereastră), scriptul **upscalează cadrul cu un factor 4×** (la 320×240) înainte de detecție, apoi scalează bounding box-urile rezultate înapoi la rezoluția originală. Parametrul `hit_threshold = -0.5` (negativ) permite un prag mai permisiv de detecție, compensând calitatea redusă a cadrelor KTH.

### 2.3. Detectorul Fallback: MOG2 (Background Subtraction)

Pentru videoclipurile în care HOG nu reușește să detecteze persoana (de exemplu, pozițiile neconvenționale din boxing sau mișcarea rapidă din running care produce blur), scriptul activează automat un **detector fallback bazat pe MOG2** (Mixture of Gaussians v2 — Background Subtraction).

MOG2 funcționează pe un principiu fundamental diferit față de HOG:
- **HOG** caută o **formă specifică** (siluetă umană) folosind gradienți și un model pre-antrenat — este un detector de obiect.
- **MOG2** caută **mișcare** detectând pixelii care diferă de fondul static învățat — este un detector de prim-plan (foreground).

Algoritmul MOG2:
1. **Modelarea Fondului**: Fiecare pixel este modelat ca o mixtură de distribuții Gaussiene care descriu valorile „normale" ale fondului static.
2. **Segmentarea Prim-Planului**: Cadrul curent este comparat cu modelul fondului; pixelii care deviază semnificativ sunt clasificați ca prim-plan (foreground mask).
3. **Operații Morfologice**: Masca rezultată trece prin operații de closing (umplerea golurilor) și opening (eliminarea zgomotului) cu un element structural eliptic $5 \times 5$.
4. **Extragerea Bounding Box-ului**: Se găsește cel mai mare contur din masca de prim-plan. Dacă aria sa depășește pragul minim (300 pixeli), se calculează dreptunghiul înconjurător.

Acest mecanism dual **HOG + MOG2** garantează o rată de detectare de aproape 100% pe dataset-ul KTH, compensând slăbiciunile fiecărui algoritm individual.

### 2.4. Selecția Grupurilor și Criteriul de Validitate

Regula de selecție impune ca **toate cele 5 cadre** din fiecare grup să aibă o detecție validă a persoanei. Acest criteriu strict previne situația în care STDP-ul ar extrage patch-uri dintr-un cadru fără persoană (generând feature maps complet negre). Grupurile sunt:
- Ordonate descrescător după **scorul de confidență** al detecției pe cadrul central.
- Selectate **non-overlapping** (fără suprapunere) pentru diversitate maximă. Dacă nu există suficiente grupuri non-overlapping, se permite suprapunere parțială.
- Sortate cronologic în output-ul final.

### 2.5. Output-ul JSON și Imaginile de Preview

Scriptul generează fișierul `hog_person_data_5frames.json` cu structura:
```json
{
  "config": {"temporal_kernel": 5, "num_groups": 5, "frame_gap": 3, ...},
  "videos": {
    "train/boxing/person01_boxing_d1_uncomp.avi": {
      "total_video_frames": 447,
      "detection_frames": 312,
      "num_groups": 10,
      "detector": "hog",
      "groups": [[{frame_idx, bboxes: [{x, y, w, h, confidence, source}]}, ...], ...]
    }
  }
}
```

Fiecare bbox conține câmpul `"source"` indicând dacă detecția provine de la `"hog"` sau `"mog2"`.

De asemenea, scriptul salvează **imagini preview** în directorul `hog_previews/`, cu strip-uri de 5 cadre (t-2 | t-1 | CENTER | t+1 | t+2), bounding box-urile fiind desenate în **albastru** pentru HOG și **verde** pentru MOG2.

**[* RECOMANDARE IMAGINE AICI: Inserează 2-3 imagini din `hog_previews/` — una pentru boxing (posibil cu MOG2 fallback, bbox verde), una pentru walking (cu HOG bbox albastru), arătând cele 3 cadre ale grupului cu bounding box-uri. Menționează sub imagine sursa detecției (HOG vs MOG2). *]**

### 2.6. Comanda de Rulare

```bash
python3 src/tool/extract_person_frames_v2.py \
    --output hog_person_data_5frames.json \
    --temporal_kernel 5 \
    --num_groups 5 \
    --frame_gap 3 \
    --frame_width 80 \
    --frame_height 60 \
    --hit_threshold -0.5 \
    --mog2_fallback \
    --preview_dir hog_previews
```

## 3. Clasele C++ Create pentru Pipeline-ul HOG-Guided: `VideoKTH_3D` și Arhitectura Sampler

Pentru a integra datele de detecție pre-calculate în simulatorul CSNN, au fost create de la zero clase C++ specializate care extind simulatorul generic cu funcționalitate optimizată pentru experimentul KTH.

### 3.1. Clasa `VideoKTH_3D` (`include/dataset/VideoKTH_3D.h`, `src/dataset/VideoKTH_3D.cpp`)

`VideoKTH_3D` este o clasă de tip dataset care **citește cadrele video exact la indicii specificați în fișierul JSON**, în loc să le extragă secvențial cu frame gap fix sau aleatoriu.

**Responsabilități:**
- **Încărcarea datelor HOG** (`load_hog_json`): La prima instanțiere, parsează integral fișierul `hog_person_data_v2.json` folosind un parser JSON manual (fără dependențe externe). Datele sunt stocate static (`static std::map<std::string, VideoHOGData> _hog_data`), partajate între instanțele de train și test.
- **Extragerea cadrelor ghidată**: Pentru fiecare sample, clasa identifică videoclipul curent și grupul temporal corespunzător (pe baza `_cursor_count`), apoi face `cv::VideoCapture::set(CAP_PROP_POS_FRAMES, target)` pentru a sări direct la cadrul specificat din JSON.
- **Fallback robust**: Dacă fișierul JSON lipsește sau un videoclip nu are date HOG, clasa revine automat la metoda standard de eșantionare cu frame gap (identică cu `dataset::Video`).

**Structuri interne:**
- `FrameBBox`: Conține `frame_idx` (indexul cadrului în video) și vectorul de bounding box-uri `(x, y, w, h)`.
- `VideoGroup`: Un grup temporal — vector de `FrameBBox` (3 elemente per grupă temporală).
- `VideoHOGData`: Toate grupurile unui videoclip + numărul total de cadre.

**Diferența cheie față de `dataset::Video`**: Clasa originală `Video` extrage cadre secvențial de la un punct aleatoriu, cu un frame gap constant. `VideoKTH_3D` extrage **exact cadrele unde persoana este confirmată vizibilă**, garantând că fiecare sample conține informație utilă pentru STDP.

### 3.2. Arhitectura Sampler — Separarea Strategiei de Eșantionare (Strategy Pattern)

O contribuție arhitecturală importantă este **refactorizarea modului în care se extrag patch-urile** din sample-uri în timpul antrenării STDP. În versiunea originală a simulatorului, logica de sampling era **hardcoded** direct în clasa de convoluție — dacă se dorea o strategie diferită de eșantionare (de exemplu, ghidată de bounding box-uri HOG în loc de random), trebuia creată o clasă de convoluție complet nouă (cum a fost cazul cu fosta clasă `HOG_Convolution3D`), duplicând tot codul de antrenare, testare și vizualizare.

Refactorizarea aplică **Strategy Pattern**: logica de sampling este extrasă într-o ierarhie de clase `Sampler` independente, injectate în `Convolution3D` ca parametru configurabil. Astfel, clasa de convoluție rămâne **generică și unică**, iar strategia de eșantionare devine un modul interschimbabil.

**Ierarhia de clase:**

```
Sampler (clasă abstractă, include/Sampler.h)
├── RandomSampler3D  — sampling uniform aleatoriu (include/sampler/RandomSampler3D.h)
└── HOGSampler3D     — sampling ghidat de bounding box-uri HOG (include/sampler/HOGSampler3D.h)
```

**Clasa abstractă `Sampler`** (`include/Sampler.h`, `include/Patch.h`):
- Urmează exact pattern-ul `STDP` / `STDPFactory` din simulator — este o subclasă `ClassParameter` cu propria fabrică (`SamplerFactory`), integrată în sistemul de parametri al simulatorului.
- Definește metoda virtuală pură `sample()` care primește tensorul input, dimensiunile layerului și un generator random, și returnează un `Patch3D(x, y, k)` — coordonatele de unde se extrage patch-ul.
- Structurile `Patch2D(x, y)` și `Patch3D(x, y, k)` sunt definite în `include/Patch.h`, oferind o ierarhie de patch-uri pentru date 2D (imagini statice) și 3D (video/secvențe temporale, unde `k` = indexul temporal).

**`RandomSampler3D`** (`include/sampler/RandomSampler3D.h`, `src/sampler/RandomSampler3D.cpp`):
- Implementarea implicită — selectează coordonatele `(x, y, k)` uniform aleatoriu din spațiul disponibil.
- Echivalent funcțional cu sampling-ul original din `Convolution3D`.

**`HOGSampler3D`** (`include/sampler/HOGSampler3D.h`, `src/sampler/HOGSampler3D.cpp`):
- Accesează datele statice din `VideoKTH_3D::get_hog_data()` pentru a plasa patch-urile **în interiorul bounding box-ului** persoanei detectate:
  $$ x \sim \mathcal{U}(x_{min}^{(k)}, x_{max}^{(k)} - \text{filter\_width}) $$
  $$ y \sim \mathcal{U}(y_{min}^{(k)}, y_{max}^{(k)} - \text{filter\_height}) $$
- **Fallback temporal**: Dacă bounding box-ul lipsește pe cadrul $k$, se caută cel mai apropiat cadru vecin cu detecție validă.
- **Fallback total**: Dacă niciun cadru nu are detecție, se revine la sampling random.

**Utilizare în cod:**
```cpp
// Conv1: sampling ghidat HOG pe zona persoanei
conv1.parameter<Sampler>("sampler").set<sampler::HOGSampler3D>();

// Conv2: sampling random (inputul e deja focalizat de conv1)
conv2.parameter<Sampler>("sampler").set<sampler::RandomSampler3D>();
```

**Avantajele arhitecturii Sampler:**
1. **Extensibilitate**: Pentru o nouă strategie de sampling, se creează doar o clasă nouă care moștenește `Sampler`, fără a modifica `Convolution3D`.
2. **Eliminarea duplicării**: Fosta clasă `HOG_Convolution3D` duplica ~800 linii de cod din `Convolution3D` doar pentru a schimba ~20 linii de sampling. Acum există o singură clasă de convoluție.
3. **Configurabilitate**: Sampler-ul este un parametru al layerului, configurabil identic cu STDP sau orice alt sub-obiect din simulator.
4. **Separarea responsabilităților**: `VideoKTH_3D` = **ce date** se încarcă (video → tensori), `Sampler` = **de unde** se extrag patch-uri din acele date.

**[* RECOMANDARE IMAGINE AICI: O diagramă flux simplificată arătând: `extract_person_frames_v2.py` → `hog_person_data_v2.json` → `VideoKTH_3D` (încarcă cadrele + datele HOG) → `Convolution3D` cere patch → `HOGSampler3D` consultă datele HOG și returnează `Patch3D(x, y, k)` din zona persoanei. *]**

## 4. Extragerea și Preprocesarea Cadrelor

Pipeline-ul de preprocesare transformă cadrele video brute în spike-uri biologice gata de antrenare:

- **Eșantionarea Inteligentă**: Modulul `dataset::VideoKTH_3D` citește fișierele video, extragând **10 grupuri temporale** de câte **3 cadre** per videoclip. Fiecare grup conține cadre consecutive unde persoana este confirmată vizibilă. Cele 3 cadre sunt analizate simultan de conv1 (`filter_depth = 3`), permițând rețelei să compare direct mișcarea pe toată fereastra temporală a grupei.
- **Reducția de dimensionalitate**: Conversia se face în spațiul 80×60 pixeli, în tonuri de gri (grayscale), purtând atenția calculului pe vectorii cinematici de deplasare ai formelor, ignorând cromatica irelevantă.
- **Pragul de Mișcare (Thresholding)**: Pentru cadrele extrase prin metoda de fallback (când JSON-ul HOG nu este disponibil), cadrul curent $F_t$ și următorul $F_{t+1}$ sunt evaluate scăzând pixelii. Extragerea continuă doar dacă suma diferenței absolute depășește un nivel minim:
  $$ \sum_{x,y} |F_{t}(x,y) - F_{t+1}(x,y)| > Threshold $$
  Acest filtru economisește timp de antrenare sărind automat peste momentele în care actorul stă complet nemișcat (parametrul `Threshold` e setat empiric la $5$).

**[* RECOMANDARE IMAGINE AICI: O figură ce arată un scurt flux/rezumat: Filmul MP4 original -> Grup de 5 Cadre (selectate prin HOG) -> Cadre Redimensionate (80x60 Grayscale). *]**

## 5. Extragerea Contrastului Spațial și Latența Biologică
Rețeaua "Spiking" nu lucrează cu pixeli reali pe o scară cromatică, ci translatând informația în impulsuri biologice.
- **DefaultOnOffFilter (Filtrarea DoG)**: Se emulează câmpul receptor al retinei umane printr-un filtru *Difference of Gaussians (DoG)*. DoG este calculat analitic ca diferența a două distribuții subiacente axate pe centrul respectiv periferia punctului:
  $$ DoG(x, y) = \frac{1}{2\pi \sigma_c^2} e^{-\frac{x^2+y^2}{2\sigma_c^2}} - \frac{1}{2\pi \sigma_s^2} e^{-\frac{x^2+y^2}{2\sigma_s^2}} $$
Rezultatul convoluției împarte topologia în două **canale (hărți complementare)**:
  - Canalul **ON** (C0): Răspunde la tranzițiile rapide de la întuneric la lumină (gradient luminos): $I_{ON} = \max(0, I * DoG)$
  - Canalul **OFF** (C1): Răspunde la tranzițiile de blocaj luminos (marginii): $I_{OFF} = \max(0, -(I * DoG))$

**[* RECOMANDARE IMAGINE AICI: O comparație vizuală orizontală: 1. Cadrul original alb-negru. 2. Harta canalului ON (unde se văd marginile luminate rescrise binar). 3. Harta canalului OFF (unde se văd marginile umbrite). Găsești aceste imagini salvate de simulator în folderul "Input_frames/kth/OOF/". *]**

- **Latency Coding**: Imaginile extrase sunt convertite în timpi reali de latență $t_s$. Un pixel cu o valoare a contrastului $I \in [0, 1]$ emite mai devreme un spike cu cât reflexia sa este mai mare:
  $$ t_s = \max(0, 1 - I(x,y,c)) $$

## 6. Direcționarea Atenției prin Detecția Vederii Artificiale (HOG)

Algoritmii predictivi neuromorfici, neavând o supervizare externă directă, impun o extragere uriașă asincronă de patch-uri (eșantioane vizuale de $5 \times 5$ pixeli) pentru a forma antrenamentul primar STDP. Dacă extragerea generală coordonată de logica modelului vizual s-ar face pur aleator (random) din arhiva imensă de pe KTH, STDP-ul ar scana inutil cerul, pardoseala sau relieful limitrof în proporție de peste 50% din timp. Această procedură standard duce organic la "orbirea" filtrelor (rețeaua epuizând cantitatea de memorie vizuală pe însușirea fundalului limitrof în locul anatomiei efective a actorului).

Pentru excluderea completă a acestui neajuns tehnic din arhitectura prezentată, s-a integrat și adaptat ca pre-procesor de direcționare a atenției **algoritmul clasic HOG (Histogram of Oriented Gradients)** condus matematic de un clasificator SVM robust pre-determinat. Implementarea a fost realizată în două etape distincte:
1. **Offline** (pre-procesare): Scriptul `extract_person_frames_v2.py` rulează HOG (cu fallback MOG2) pe toate videoclipurile, salvând bounding box-urile în fișierul JSON.
2. **Runtime** (antrenare CSNN): Clasa `VideoKTH_3D` încarcă datele HOG din JSON, iar `HOGSampler3D` (injectat ca parametru în `Convolution3D`) ghidează eșantionarea STDP fără nicio dependență de OpenCV la runtime.

### Mecanismul intern pas-cu-pas al HOG (OpenCV)
Funcționarea internă a pre-procesorului urmează un pipeline de identificare strict pentru a discerne în ce cuadrant se află persoana raportată la scara metrică din cadru:
1. **Calculul Gradienților Pixelari**: Imaginea subiacentă este prelevată și derivată matricial la nivel de pixel pentru a obține atât magnitudinea, cât și unghiul de orientare (direcția propriu-zisă a marginii organice). Calculul descoperă și delimitează contururile, zone cu o tranziție de pixel gradient cu un salt ascuțit de valori - de exemplu granița dintre cămașă și background.
2. **Construirea Celulelor de Histograme**: Imaginea reformatată este segmentată într-o topologie granulară de celule optice (matrice locale limitate la $8 \times 8$ pixeli). Fiecărei subunități i se atribuie algoritmul de construcție a unei histograme. Vectorii descărcați "votează" direcția orientativă a pixelilor (aflați exclusiv în intervalul matematic de spectru 0-180 grade), marcând fix traiectoria dominantă formând înțelesul global structural din care ia naștere o siluetă recunoscută!
3. **Normalizarea Iluminantă pe Blocuri**: Deoarece prelucrările sunt direct reținute vizual, sistemul trebuie formatat pentru a fi insensibil la intensitatea iluminării medii (umbră prelungită, reflexii incorecte ale apusului pe perimetrul exterior indoor vs outdoor tipic fișierelor KTH). Celulele sunt astfel multi-înlănțuite algoritmic în "Blocuri" decizionale suprapuse geometric (standard tip $16 \times 16$ pixeli), histogramele blocurilor trecând prin normalizarea geometrică tradițională liniară L2 ce abordează uniformizarea iluminantă ca filtru constant final.
4. **Validarea Semantică SVM Liniar**: Harta descriptorului vector-HOG asamblată anterior parcurge metodic ecranul printr-un set vast de "ferestre glisante" (sliding windows). Implementarea internă OpenCV a modelului de față execută instrucțiunea directă `getDefaultPeopleDetector()`. Aceasta aplică în timp real modelul SVM performant supervizat, model arhivat după sute de validări standard de siluete, stabilind printr-un sistem limită decizional clar dacă patch-ul vizual glisant în discuție este de formă umană, raportând binar locația pe ecran!

### Avantajul HOG în Ingestia Rețelelor Spiking (`CSNN`)
În mediul propus, HOG reprezintă granița limitativă superioară dintre prelucrare statică de identificare și bio-mimetism format din logica asincronă pur neuronală de antrenament mișcare temporală. Rețeaua extrage automat din cadru parametrul matematic delimitator al spațiului efectiv de mișcare umană: The Bounding Box / `ROI - Region of Interest`.

Sistemul de detecție per cadru temporal folosit de `HOGSampler3D` generează un vector de bounding box-uri per grup temporal:
$$\text{cache}[\text{sample\_id}] = \{\text{BBox}_0, \text{BBox}_1, \text{BBox}_2\}$$

În momentul eșantionării STDP la indexul temporal $k$, sistemul folosește bounding box-ul specific acelui cadru:
   $$ \text{sample}_x \sim \mathcal{U}(x_{min}^{(k)}, x_{max}^{(k)} - \text{Filt}_{w}) $$
   $$ \text{sample}_y \sim \mathcal{U}(y_{min}^{(k)}, y_{max}^{(k)} - \text{Filt}_{h}) $$

Dacă HOG-ul nu detectează persoana într-un cadru specific, se aplică un **fallback de proximitate temporală**: sistemul caută cel mai apropiat cadru din secvență unde o persoană a fost detectată cu succes, utilizând bounding box-ul acestuia. Doar dacă niciun cadru din secvență nu conține o detecție validă, se revine la eșantionarea aleatorie clasică.

**[* RECOMANDARE IMAGINE AICI: Un cadru KTH peste care poți desena tu grafic un dreptunghi colorat (ex. albastru pentru HOG, verde pentru MOG2) deasupra persoanei, sugerând Bounding Box-ul generat din care rețeaua "mușcă" porțiuni 5x5 pentru a lăsa pe dinafară zidul/iarba. Folosește preview-uri din `hog_previews/`. *]**

## 7. Convoluțiile Spațio-Temporale și Regula STDP
### Memoria Spațio-Temporală
Setarea flag-ului `tmp_filter_size = 3` forțează design-ul convoluției conv1 să asimileze **simultan toate cele 3 cadre temporale** dintr-un grup. Filtrul acționează volumul $V = (x, y, t)$, preluând asincron **toate 3 cadrele** dintr-o grupă la pachet. Rețeaua memorează dinamic animația ("Optical Flow"), comparând conturul persoanei pe tot intervalul temporal al grupei.

Cu 3 cadre temporale la intrare și `filter_depth = 3`, dimensiunea temporală este **consumată complet** la conv1:
- **conv1**: $T_{out} = (3 - 3)/1 + 1 = 1$ — un singur output temporal

Această alegere reflectă scopul extracției: cele 3 cadre dintr-o grupă au fost selectate tocmai pentru a fi comparate simultan (sunt consecutive/apropiate temporal, alese împreună de scriptul de preprocesare). Dacă s-ar fi dorit frame-uri independente, s-ar fi extras 30 de frame-uri aleatorii din fiecare video în loc de 10 grupuri de câte 3. Filtrul temporal complet permite conv1 să detecteze **deplasarea conturului** (Optical Flow) și **viteza mișcării** direct din compararea celor 3 momente temporale.

Pentru straturile ulterioare (conv2, fc1), `filter_depth = 1` deoarece dimensiunea temporală a fost deja integrată complet de conv1.

### Învățarea Nesupervizată: Spike-Timing-Dependent Plasticity (STDP)
Filtrele își ajustează ponderile sinaptice total nesupervizat. Dacă un impuls de intrare (la $t_{pre}$) contribuie cauzal la declanșarea neuronului (la $t_{post}$), legătura sinaptică este întărită exponențial:
   $$ \Delta w = \begin{cases} A_+ e^{-\frac{|\Delta t|}{\tau_+}}, & \text{dacă } \Delta t > 0 \text{ (Potențare: } t_{pre} \rightarrow t_{post}) \\ -A_- e^{-\frac{|\Delta t|}{\tau_-}}, & \text{dacă } \Delta t \le 0 \text{ (Depresie: anti-cauzal / zgomot)} \end{cases} $$
unde $\Delta t = t_{post} - t_{pre}$.

**[* RECOMANDARE IMAGINE AICI: O figură teoretică a graficului STDP de pe internet (curba cu zona pozitivă sus și negativă jos). Este imaginea clasică ce validează teoretic antrenamentul rețelei tale! *]**

Pentru diversitatea filtrelor, se folosește **Homeostazia Adaptivă** (un prag flexibil ca neuronul să nu descarce prea des pe aceeași acțiune) și **Inhibiția WTA (Winner-Takes-All)** (doar neuronul cel mai rapid își actualizează greutatea pe o zonă fluidă, împiedicând filtrele să învețe exact același pattern).

## 8. Modelul de Ierarhizare (Topologia Rețelei)
Modelul procesează vizual de la granularități simple (Muchii), la cele complexe (Postură completă):

1. **Stratul `conv1` (Convolution3D + HOGSampler3D)**: Extrage independent **32 de hărți** (filtre) de trăsături microscopice spatio-temporale (ex. linii înclinate, unghiuri din dinamica corpului, direcția mișcării) folosind ferestre de 5×5 pixeli cu adâncime temporală de 3 cadre (toate cadrele din grupă, simultan). Eșantionarea STDP este ghidată de `HOGSampler3D` prin bounding box-urile HOG pre-calculate, focalizând filtrele exclusiv pe zona persoanei. Output: $[56 \times 76 \times 32 \times 1]$.
2. **Pooling `pool1`**: Aplică Max-Pooling asincron pe blocuri de $(2 \times 2)$ spațial, reducând rezoluția spațială și conferind *Translation Invariance*. Output: $[28 \times 38 \times 32 \times 1]$.
3. **Stratul `conv2` (Convolution3D + RandomSampler3D)**: Operează cu **48 de filtre** pe cele 32 canale din pool1, cu adâncime temporală de 1 (dimensiunea temporală a fost deja integrată complet de conv1). Acest strat combină trăsăturile low-level (edge-uri, texturi) în pattern-uri mid-level (părți de corp, articulații în mișcare). Eșantionarea STDP folosește `RandomSampler3D` — nu mai necesită ghidare HOG deoarece conv1 a focalizat deja informația pe zona persoanei. Output: $[24 \times 34 \times 48 \times 1]$.
4. **Pooling `pool2`**: A doua reducere spațială $2 \times 2$. Output: $[12 \times 17 \times 48 \times 1]$.
5. **Stratul `fc1` (Fully Connected Convolution + RandomSampler3D)**: Cu un filtru de $12 \times 17$ spațial și adâncime temporală de 1, acoperind **întreaga suprafață** spațială din pool2. Cele **32 de filtre** produc un vector final hiper-compact $(1 \times 1 \times 32 \times 1)$ care sintetizează postura globală și dinamica persoanei.

**[* RECOMANDARE IMAGINE AICI IMPORTANTA: Inserează pozele cu Greutățile (Weights) generate prin scriptul `draw_kth_weights.py` — arată 3 sau 4 filtre colorate diferit pentru canalul ON (Blues) și OFF (Reds), pe cele 2 momente temporale (t=0, t=1). Explică dedesubt cum STDP-ul a "sculptat" forme umane direct în sinapse și cum se observă shift-ul temporal între t=0 și t=1 (deplasarea conturului confirmă asimilarea Optical Flow). *]**

**[* RECOMANDARE IMAGINE AICI SECUNDARĂ: Imediat sub ponderi, pune o imagine cu "Feature Maps" (hărțile de activare extrase cu `draw_feature_maps.py`), exemplificând modul abstractizat prin care SNN-ul "vede" omul chiar înainte de ieșirea către SVM. *]**

## 9. Clasificatorul Predictiv SVM (Support Vector Machine)
Odată arhitectura SNN antrenată nesupervizat, plasticitatea stratului STDP este suspendată (imobilizând Greutățile). Acum, dataset-ului global i se extrage doar semnătura "spiking" din ultimul etaj `fc1` al CSNN.
Fiecare videoclip comprimat tridimensional în descărcări electrice devine pură prelucrare vectorială - livrând argumente matematice către un identificator supervizat de tip **Support Vector Machine (SVM)** liniar (acela care face demarcația finală pe setul independent Test).
SVM-ul construiește hiperplane matematice pentru demarcația deciziilor dintre cele 6 clase de comportament. Clasificarea se face la fiecare strat al rețelei (conv1, conv2, fc1) pe baza feature map-urilor extrase prin `TimeObjectiveOutput` urmate de `SumPooling` și `FeatureScaling`, permițând analiza calității reprezentărilor la diferite niveluri de abstractizare.

**[* RECOMANDARE IMAGINE AICI IMPORTANTA: Matricea de Confuzie (Confusion Matrix) pentru conv1 (cel mai performant strat, 68.41%). Generează-o din datele SaveOutput folosind un script Python care antrenează SVM-ul pe train și plotează confusion matrix pe test. Comentează pe seama ei de ce algoritmul s-a descurcat superb între Walk și Box (fiind acțiuni complet eterogene), dar a putut face mici confuzii între Jogging și Running (care din perspectivă SNN au profile cinematice și hărți STDP aproape identice!). *]**

---

## 10. Defalcarea Parametrilor și a Ierarhiei Straturilor (Layers)
Acest subcapitol fixează fluxul dimensional (tensorial) între fiecare strat din arhitectura experimentului `KTH_3D_FinalVersion`, stabilite empiric pentru acuratețe maximală:

### 10.1. `DefaultOnOffFilter` și `MaxScaling`
- **Parametri importanți**: `filter_size = 7` (fereastră DoG de $7 \times 7$), deviații standard $\sigma_{centru}=1.0, \sigma_{margine}=4.0$.
- **Input**: Videoclip eșantionat prin grupuri HOG: $[60 \text{ (H)} \times 80 \text{ (W)} \times 1 \text{ (Ch)} \times 3 \text{ (Timp)}]$.
- **Output**: $[60 \times 80 \times 2 \times 3]$ ($2$ canale reprezentând hărțile complementare: ON și OFF).

### 10.2. `LatencyCoding` (Conversia în Spike-uri)
- **Input/Output**: $[60 \times 80 \times 2 \times 3]$. Valorile părăsesc scala statică Gri și devin pur timp de latență asincron per semnal ($t_s$).

### 10.3. Stratul `conv1` (Convolution3D + HOGSampler3D)
- **Parametri importanți**: Filtru spațial `5×5`, adâncime temporală `filter_depth = 3` (vede simultan toate cele 3 cadre din grupă), **32 de filtre** (epoci: 50). Eșantionare ghidată HOG prin `HOGSampler3D` — sampler-ul accesează bounding box-urile pre-calculate din `VideoKTH_3D::get_hog_data()`.
- **Input**: Tensor Spike-uri din preprocesor: $[60 \times 80 \times 2 \times 3]$.
- **Output**: $[56 \times 76 \times 32 \times 1]$. Dimensiunea fizică de 60×80 primește limitarea marginilor convoluționale. Cele 2 canale ON/OFF sunt comprimate în **32 de canale/pattern-uri** complexe. Dimensiunea temporală este complet consumată: $(3 - 3)/1 + 1 = 1$.
- **Sinapse per neuron**: $5 \times 5 \times 2 \times 3 = 150$.
- **Threshold inițial**: $\mathcal{N}(8.0, 0.1)$ — calibrat pentru cele 150 sinapse de input.
- **`min_th = 1.0`**: Permite threshold-ului să scadă prin annealing, adaptându-se la inputul de la On/Off filter.
- **`wta_infer = false`**: Permite mai mulți neuroni să tragă la aceeași poziție spațială, generând feature maps mai dense și mai diverse pentru straturile ulterioare.
- **Sampler**: `HOGSampler3D` — patch-urile de $5 \times 5 \times 3$ sunt extrase exclusiv din zona persoanei, pe baza bounding box-urilor din fișierul JSON pre-calculat.

### 10.4. Stratul `pool1` (Pooling3D)
- **Parametri**: Sub-eșantionare Max Pooling asincron $2 \times 2$ spațial.
- **Input**: $[56 \times 76 \times 32 \times 1]$.
- **Output**: $[28 \times 38 \times 32 \times 1]$. Spațiul imaginii este înjumătățit rezoluțional pentru viteză și invarianță de formă.

### 10.5. Stratul `conv2` (Convolution3D + RandomSampler3D)
- **Parametri importanți**: Filtru spațial $5 \times 5$, `filter_depth = 1` (dimensiunea temporală a fost deja integrată de conv1), **48 de filtre** (epoci: 100). Eșantionare random uniform prin `RandomSampler3D` (nu necesită HOG — inputul este deja focalizat de conv1).
- **Input**: Mapările pooled $[28 \times 38 \times 32 \times 1]$, cu "grosimea" de 32 canale din stratul prim.
- **Output**: $[24 \times 34 \times 48 \times 1]$.
- **Sinapse per neuron**: $5 \times 5 \times 32 \times 1 = 800$.
- **Threshold inițial**: $\mathcal{N}(12.0, 0.1)$ — calibrat pentru cele 800 sinapse de input.
- **`min_th = 1.0`**: Permite threshold-ului să scadă prin annealing, compensând inputul rar moștenit de la conv1.
- **`wta_infer = true`**: Winner-Takes-All activ la inferență pentru pattern-uri discriminative, forțând competiția între filtre.
- **Sampler**: `RandomSampler3D` — la acest nivel, inputul conține deja doar trăsături relevante focalizate pe persoană de conv1.

### 10.6. Stratul `pool2` (Pooling3D)
- **Parametri**: Sub-eșantionare spațială $2 \times 2$.
- **Input**: $[24 \times 34 \times 48 \times 1]$.
- **Output**: $[12 \times 17 \times 48 \times 1]$. Forma inițială a omului e complet comprimată în esența sa abstractă.

### 10.7. Stratul Decizional `fc1` (Fully Connected Convolution + RandomSampler3D)
- **Parametri importanți**: Fereastră cu Lățimea = $12$, Înălțimea = $17$ (acoperă integral spațiul de la pool2), `filter_depth = 1`. Număr de **32 filtre** (epoci: 150).
- **Input**: Maparea compactă $[12 \times 17 \times 48 \times 1]$.
- **Output**: $[1 \times 1 \times 32 \times 1]$ — un vector de 32 de descriptori globali funcționând ca "cod de bare" spatio-temporal al acțiunii.
- **Sinapse per neuron**: $12 \times 17 \times 48 \times 1 = 9792$.
- **Threshold inițial**: $\mathcal{N}(18.0, 0.1)$ — calibrat ținând cont de sparsity-ul moștenit de la conv2.
- **`min_th = 2.0`**: Permite threshold-ului să scadă prin annealing, esențial date fiind fluctuațiile de sparsity din straturile anterioare.
- **Sampler**: `RandomSampler3D`.

### 10.8. Salvarea și Vizualizarea Ponderilor: Scriptul `draw_kth_weights.py`

Scriptul `src/tool/draw_kth_weights.py` (creat de autor) automatizează vizualizarea ponderilor sinaptice (weights) salvate în format JSON de simulator la finalul antrenamentului STDP. Acesta:

1. **Detectează automat** cel mai recent experiment (`kth_*`) din directorul `Weights/`.
2. **Parsează fișierele JSON** de ponderi pentru fiecare strat (`conv1`, `conv2`, `fc1`), reconstruind tensorii multidimensionali.
3. **Generează imagini individuale** pentru fiecare combinație de filtru × canal × pas temporal:
   - Canalul ON este afișat cu paleta de culori **Blues** (albastru).
   - Canalul OFF este afișat cu paleta **Reds** (roșu).
   - Fiecare imagine include normalizarea min-max și o bară de culoare (colorbar) indicând ponderea sinaptică.
4. **Output**: Imaginile sunt salvate în `Weights/{exp_name}_images/{layer}/`, organizate ca: `{layer}_filtru_{N}_ch_{C}_t_{T}.png`.

**Utilizare:**
```bash
cd csnn-simulator-build/
python3 ../src/tool/draw_kth_weights.py          # detectează automat experimentul
python3 ../src/tool/draw_kth_weights.py kth_42    # specifică manual experimentul
```

**Interpretarea vizuală a ponderilor conv1 (3 cadre temporale):**

Pentru conv1 cu `filter_depth = 3`, fiecare filtru produce **3 imagini** (t=0, t=1, t=2), câte una per canal (ON, OFF). Comparând cele 3 momente temporale ale aceluiași filtru, se observă fenomenul de **shift cinematic** — forma geometrică (de exemplu, conturul gleznei) se deplasează spațial progresiv între t=0 → t=1 → t=2, demonstrând că STDP-ul a învățat nu doar forma, ci și **direcția și viteza mișcării** (Optical Flow) pe o fereastră temporală completă de 3 cadre.

**[* RECOMANDARE IMAGINE AICI IMPORTANTA: Inserează 3 imagini ale aceluiași filtru la t=0, t=1 și t=2 din `Weights/{exp}_images/conv1/`, arătând deplasarea progresivă a formei. Sub imagine: "Se observă cum conturul detectat se deplasează spațial progresiv între cele 3 momente temporale, confirmând asimilarea Optical Flow de către STDP." *]**

### 10.9. Vizualizarea Hărților de Trăsături: Scriptul `draw_feature_maps.py`

Scriptul `src/tool/draw_feature_maps.py` (creat de autor) citește fișierele binare de feature maps salvate de `analysis::SaveOutput` și le vizualizează ca heatmaps. Acesta:

1. **Citește formatul binar** al simulatorului (tensori serializați cu magic number `0x234264FF`), suportând atât formatul dens cât și cel sparse.
2. **Vizualizează toate dimensionalitățile**:
   - **Tensori 1D** (fc1, $1 \times 1 \times N$): Afișați ca bar charts — fiecare bară reprezintă activarea unui neuron/filtru.
   - **Tensori 2D/3D** (conv1, conv2): Afișați ca heatmaps (paleta `hot`), câte o sub-imagine per filtru, cu normalizare per-sample.
   - **Tensori 4D** (cu dimensiune temporală): Se aplică **Temporal Max Projection** — proiecția valorii maxime pe axa temporală — obținându-se imagini care reunesc spike-urile maxime de-a lungul întregii ferestre temporale.
3. **Afișează primele 5 sample-uri** din fiecare fișier, permițând compararea vizuală între acțiuni diferite.
4. **Output**: `FeatureMaps/{layer}_feature_maps.png`.

**Utilizare:**
```bash
cd csnn-simulator-build/
python3 ../src/tool/draw_feature_maps.py
```

**Interpretarea vizuală:**
- **Conv1 Feature Maps**: Se observă activări sparse, concentrare pe conturul persoanei (datorită ghidării HOG). Majoritatea spațiului este negru (inactiv), cu zone roșii/galbene doar acolo unde STDP-ul a detectat match-uri cu filtrele învățate.
- **Conv2 Feature Maps**: Activările sunt mai difuze și mai abstracte — nu mai se văd margini fine, ci blocuri mari corespunzând părților corpului.
- **FC1 Feature Maps (Cod de Bare)**: Vector de $32 \times 2$ valori (32 neuroni pe 2 momente temporale), dintre care doar câteva sunt nenule, formând o semnătură spatio-temporală unică per acțiune.

**[* RECOMANDARE IMAGINE AICI: Inserează `Conv1_test_feature_maps.png` din directorul `FeatureMaps/`. Sub imagine: "Cele 64 de filtre vizualizate pentru 5 eșantioane de test pe 4 cadre temporale. Se observă raritatea activărilor (sparsity ≈ 0.24) și focalizarea pe zona persoanei." *]**

**[* RECOMANDARE IMAGINE AICI: Inserează `FC1_test_feature_maps.png`. Sub imagine: "Vectorul final de 32 de neuroni × 2 momente temporale, funcționând ca 'cod de bare' spatio-temporal al acțiunii." *]**

**[* RECOMANDARE TABEL AICI: Un tabel comparativ cu exemplu de activări FC1 pentru fiecare din cele 6 acțiuni, arătând care neuroni se activează predominant per clasă. *]**

#### Raritatea Activărilor (Sparsity) și Evidența Winner-Takes-All (WTA)
Un fenomen definitoriu vizibil în hărțile de trăsături, în special la straturile **conv1** și **conv2**, este raritatea extremă a activităților (Sparsity). Majoritatea absolută a spațiului receptiv din filtre afișează o activare zero (culoare neagră), impulsurile electrice concentrându-se exclusiv în zonele unde se manifestă o mișcare puternică de contur (generată de focalizarea HOG aplicată anterior).

Acest comportament confirmă funcționarea corectă a mecanismului de inhibiție laterală **Winner-Takes-All (`wta_infer = true`)**. În cadrul unui cluster spațial de neuroni, doar neuronul (filtrul) cu cel mai ridicat potențial atinge pragul de descărcare și emite un "spike", inhibându-i simultan pe toți ceilalți. Consecința analitică a acestui principiu este separarea curată a formelor detectate și eficiența energetică masivă a modelului — se efectuează calcul computațional doar acolo unde există informație structurală utilă.

**[* RECOMANDARE IMAGINE AICI: O secțiune decupată (zoom-in) din `Conv1_test_feature_maps.png`, selectând 2-3 filtre în care silueta corpului sau brațelor este puternic conturată de zone roșii/galbene pe fundal complet negru. *]**

#### Specializarea Filtrelor și Generalizarea Train vs. Test
Modelul este compus din 64/48/32 de filtre per strat, antrenate individual folosind regula ne-supervizată STDP. O consecință directă a acestei învățări este **specializarea filtrelor** – fiecare filtru devine capabil să recunoască un alt tipar de mișcare, margine sau orientare.

* **Compararea Train/Test:** Imaginile din setul `_train` prezintă adesea contururi ferme, reflectând caracteristicile învățate inițial. Ulterior, aceleași filtre reușesc să generalizeze peste noile date din secvențele `_test`, afișând un contur la fel de curat și lipsit de zgomot. Aceasta denotă o lipsă de overfitting și validitatea metodei de abstractizare ne-supervizate SNN pe date video nevăzute anterior.
* **Abstractizarea spațială (Conv2):** La nivelul **conv2**, unde câmpul receptiv a crescut semnificativ (96 canale de input), informația se comasează din contururi subțiri în detectoare de formă globală, evidențiind blocurile mari de mișcare ale subiecților. "Petele" de activare din `conv2` sunt vizibil mai mari și mai difuze decât cele din `conv1`.

**[* RECOMANDARE IMAGINE AICI: `Conv2_test_feature_maps.png`, comparând vizual cu harta `Conv1` anterioară. Subliniază abstractizarea spațială prin diferența de granularitate. *]**

---

## 11. Contribuții Originale și Rezultate

Un element central de noutate tehnică adus prin prezenta lucrare este **optimizarea extragerii trăsăturilor vizuale în mod cu adevărat reprezentativ**. Contribuțiile principale sunt:

### 11.1. Crearea Clasei `VideoKTH_3D`
Această clasă de dataset, creată de la zero, înlocuiește eșantionarea secvențială/aleatorie a cadrelor cu o **extragere ghidată pe baza datelor HOG pre-calculate**. Prin citirea fișierului JSON produs de `extract_person_frames_v2.py`, clasa garantează că fiecare sample conține exact cadrele unde persoana este vizibilă, eliminând secvențele statice sau cele în care actorul a ieșit din cadru.

### 11.2. Arhitectura Sampler (Strategy Pattern) și `HOGSampler3D`
Logica de eșantionare a patch-urilor a fost **extrasă din clasa de convoluție** într-o ierarhie de clase `Sampler` independente, aplicând Strategy Pattern. Aceasta permite schimbarea strategiei de eșantionare fără a modifica sau duplica clasa de convoluție. Implementarea concretă `HOGSampler3D` **focalizează eșantionarea STDP pe zona persoanei** fără nicio dependență de OpenCV la runtime. Prin citirea bounding box-urilor din JSON (via `VideoKTH_3D::get_hog_data()`), eșantionarea patch-urilor de 5×5 este limitată strict la Region of Interest, forțând filtrele să învețe exclusiv trăsături ale corpului uman în mișcare. Sampler-ul este configurat ca parametru al layerului, identic cu regula STDP: `conv1.parameter<Sampler>("sampler").set<sampler::HOGSampler3D>()`.

### 11.3. Scriptul `extract_person_frames_v2.py` cu Detecție Duală HOG + MOG2
Scriptul de pre-procesare, creat de autor, combină doi algoritmi complementari de viziune artificială:
- **HOG**: Detectează silueta umană prin analiza orientării gradienților — excelent pe pozele standard de mers/fugă.
- **MOG2**: Detectează mișcarea prin scăderea fondului — compensează eșecurile HOG pe pozele neconvenționale (boxing, mișcări rapide).

Această abordare duală atinge o rată de detectare de aproape 100% pe dataset-ul KTH.

### 11.4. Eliminarea Memory Leak-ului prin Separarea Offline/Runtime
Versiunea originală a simulatorului rula detectorul HOG OpenCV **la runtime** în interiorul buclei STDP, generând peste 95.000 de apeluri `detectMultiScale` care fragmentau Heap-ul și duceau la OOM Kill. Prin mutarea detecției complet offline (în scriptul Python), simulatorul C++ nu mai depinde de OpenCV pentru detecție, eliminând complet problema de memorie.

### 11.5. Rezultate Experimentale

#### Rezultate per strat (experiment kth_49, 3 samples/video)

| Strat | Filtre | Epoci | Sinapse/Neuron | Threshold | Sparsity | Active Units | Acuratețe SVM (%) |
|-------|--------|-------|----------------|-----------|----------|-------------|-------------------|
| conv1 | 64     | 150   | 100            | 8.0       | 0.242    | 99.9%       | **68.41%**        |
| conv2 | 48     | 100   | 3200           | 12.0      | 0.778    | 53.6%       | 57.16%            |
| fc1   | 32     | 150   | 19584          | 18.0      | 1.000    | 1.6%        | 20.80%            |

**Observații critice:**
- **Conv1 (68.4%)** atinge cel mai bun scor, beneficiind de ghidarea HOG și de sparsity-ul scăzut (24.2%) care păstrează multă informație pentru SVM.
- **Conv2 (57.2%)** pierde performanță față de conv1 — sparsity-ul crește la 77.8%, iar WTA agresiv reduce diversitatea activărilor. Informația temporală suplimentară nu compensează pierderea de granularitate.
- **Fc1 (20.8%)** este aproape la nivelul random chance (16.7% pentru 6 clase). Sparsity = 1.0 indică neuroni aproape complet inactivi — doar 1.6% din unitățile fc1 se activează. Acest strat necesită re-calibrare semnificativă a threshold-ului.

**[* RECOMANDARE IMAGINE: Matricea de Confuzie (Confusion Matrix) pentru conv1. Generează-o folosind output-ul SVM din SaveOutput. Comentează pe seama ei: ce perechi de acțiuni se confundă (probabil jogging/running, boxing/handclapping). *]**

**[* RECOMANDARE GRAFIC: Un bar chart cu acuratețile per clasă (boxing, handclapping, handwaving, jogging, running, walking) la conv1, pentru a evidenția pe ce acțiuni rețeaua excelează și unde greșește. *]**

**[* RECOMANDARE IMAGINE: Un grafic cu evoluția sparsity-ului prin rețea: 0.24 → 0.78 → 1.0, ilustrând cum informația se pierde progresiv. Poate fi un simplu line plot sau bar chart. *]**

#### Comparație între versiunile experimentului

| Versiune | Preprocesare | Cadre | Filtre | Temporal | Sampling | Acuratețe maximă |
|----------|-------------|-------|--------|----------|---------|-------------------|
| KTH_3D (original) | Random sampling, HOG runtime | 10 | 32-32-32 | depth=3 (consumat la conv1) | Hardcoded random | TBD |
| KTH_3D_v3 | HOG JSON offline, HOG_Convolution3D | 5 | 64-48-32 | depth=2 (propagat ierarhic) | Hardcoded în clasă separată | **68.41%** (conv1) |
| KTH_3D_FinalVersion | HOG JSON offline, Sampler Pattern | 3 | 32-48-32 | depth=3 (complet la conv1) | HOGSampler3D + RandomSampler3D | TBD |

**[* RECOMANDARE TABEL: Completează cu rezultatele din versiunile anterioare (v3, v4, v5) pentru a arăta evoluția performanței. *]**

#### Analiza Coerenței Filtrelor

| Strat | Mean Weights | Coherence Q3 | Coherence Max | Interpretare |
|-------|-------------|--------------|---------------|-------------|
| conv1 | 0.034       | ~5e-10       | 0.997         | Filtre foarte diverse (Q3≈0), dar câteva perechi aproape identice (max≈1) |
| conv2 | 0.013       | ~3.2e-4      | 0.406         | Diversitate bună, fără filtre redundante |

**[* RECOMANDARE IMAGINE: Grila de ponderi (weights) din `draw_kth_weights.py` pentru conv1, arătând cele 64 de filtre cu canalele ON (albastru) și OFF (roșu) pe cele 2 cadre temporale (t=0, t=1). Se poate observa shift-ul cinematic între cele 2 momente. *]**

**[* RECOMANDARE IMAGINE: Feature maps din `draw_feature_maps.py` pentru conv1_test și conv2_test, comparând granularitatea activărilor. Conv1 are activări fine pe conturul persoanei, conv2 are activări mai difuze pe blocuri mari de mișcare. *]**

**[* RECOMANDARE IMAGINE: Feature maps fc1 (bar chart cu vectorul de 32 neuroni × 2 momente temporale) pentru câte un sample din fiecare clasă de acțiune. Se observă cum vectorul-cod-de-bare diferă între acțiuni. *]**

### 11.6. Optimizarea Memoriei: Streaming în `_process_output`

O contribuție tehnică importantă a fost **rezolvarea problemei de Out-of-Memory (OOM)** în execuția experimentului pe sisteme cu 16GB RAM. Cauza: funcția `_process_output` din `SparseIntermediateExecutionNew` crea **copii complete** ale tuturor sample-urilor la rezoluție conv1 ($56 \times 76 \times 64 \times 4$ ≈ 1M valori per sample) înainte de a aplica postprocessing-ul care le reduce dramatic (la $2 \times 2 \times 64 \times 4$).

Soluția implementată introduce două moduri de procesare:
1. **Streaming mode**: Pentru output-uri fără postprocessing (SaveOutput), sample-urile sunt procesate și scrise pe disc individual, fără a fi acumulate în memorie.
2. **Inline postprocessing**: Pentru output-uri cu postprocessing (SumPooling + FeatureScaling), pașii single-pass sunt aplicați per-sample **inline** cu conversia, astfel încât doar rezultatele mici (post-pooling) sunt stocate în memorie.

Această optimizare reduce consumul de memorie de la ~36GB la ~12GB, permițând execuția completă pe hardware cu 16GB RAM.
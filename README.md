# Sample Gap Filler s Quality-Aware Algoritmem

## Popis
Tento program doplňuje chybějící audio vzorky hudebních nástrojů transpozicí z nejbližších dostupných vzorků. 
Používá quality-aware algoritmus založený na Paretově pravidle: 
analyzuje kvalitu not podle počtu vzorků (více vzorků = lepší kvalita) a transponuje pouze z kvalitních zdrojů. 
Navazuje na nástroj Pitch Corrector a sdílí principy zpracování audio (simple pitch shifting pomocí resample).

Klíčové funkce:
- Skenování existujících WAV souborů v specifickém formátu: `m{midi:03d}-vel{velocity}-f{44|48}[-next{N}].wav`.
- Analýza kvality: Vyřadí spodních X% not s nejmenším počtem vzorků (např. 20% při threshold 0.8).
- Analýza pokrytí MIDI rozsahu 21-108 (A0-C8) pro velocity layers 0-7 a sample rates 44.1kHz/48kHz.
- Generování chybějících vzorků transpozicí (±1 až ±3 půltóny, priorita dolů).
- Kopírování kvalitních originálních vzorků do výstupu.
- Report chybějících/vyloučených vzorků v `missing-notes.txt`.

## Požadavky
- Python 3.12+
- Knihovny: `soundfile`, `numpy`, `resampy`, `tqdm`, `re`, `sys`, `shutil`, `argparse`, `collections`, `pathlib`.
- Žádný internet – vše offline.

## Instalace
1. Nainstalujte Python 3.12.
2. Nainstalujte závislosti příkazem:
   ```
   pip install soundfile numpy resampy tqdm
   ```
   (Ostatní knihovny jsou standardní.)

3. Stáhněte nebo zkopírujte skript `sample_gap_filler.py`.

## Použití
Spusťte skript s argumenty přes příkazovou řádku:

```
python sample_gap_filler.py --input-dir CESTA_K_VSTUPU --output-dir CESTA_K_VYSTUPU [volitelné argumenty]
```

Příklad:
```
python sample_gap_filler.py --input-dir ./processed_samples --output-dir ./complete_samples --quality-threshold 0.8 --verbose
```

### Argumenty
- `--input-dir` (povinný): Cesta k adresáři s existujícími vzorky (výstup z Pitch Corrector).
- `--output-dir` (povinný): Cesta k výstupnímu adresáři (bude vytvořen, pokud neexistuje).
- `--quality-threshold` (volitelný, default: 0.8): Prah kvality (0.1-1.0) – ponechá top X% not podle počtu vzorků.
- `--verbose` (volitelný): Podrobný výstup (vypne progress bary).

Program projde fázemi: skenování, analýza kvality, analýza pokrytí, generování, kopírování. Výstup obsahuje shrnutí a report.

## Konfigurace
- MIDI rozsah: 21-108 (nelze měnit bez úpravy kódu).
- Max transpozice: ±3 půltóny.
- Cílové sample rates: 44.1kHz a 48kHz.
- Pokud potřebujete změny (např. jiné rozsahy), upravte konstanty v třídě `SampleGapFiller`.

## Výstup
- Doplněné vzorky v `--output-dir` ve formátu `m{midi:03d}-vel{velocity}-f{44|48}.wav`.
- Report `missing-notes.txt`: Seznam vzorků, které se nepodařilo vygenerovat nebo byly vyloučeny kvůli kvalitě.
- Konzolový výstup: Statistiky (např. vygenerováno X vzorků, vyloučeno Y).

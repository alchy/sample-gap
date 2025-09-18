# Sample Gap Filler

Program pro doplňování chybějících zvukových vzorků hudebních nástrojů transpozicí z nejbližších dostupných vzorků.
Navazuje na Pitch Corrector a používá stejné principy zpracování audio dat. Určen pro IthacaSampler.

## Popis

- **Funkce**:
  - Skenuje existující WAV soubory v input_dir (očekávaný formát: `m{MIDI:03d}-vel{velocity}-f{44|48}.wav`, např. `m060-vel4-f44.wav` pro 44.1 kHz nebo `m062-vel7-f48.wav` pro 48 kHz).
  - Analyzuje pokrytí MIDI rozsahu 21–108 (A0–C8) pro velocity layers 0–7 a sample rates 44.1 kHz / 48 kHz.
  - Doplňuje chybějící vzorky transpozicí (pitch shift) z nejbližších existujících (±3 půltóny max), s prioritou směru dolů.
  - Generuje výstup do output_dir ve stejném formátu.
  - Ukládá report nedoplňitelných vzorků do `missing-notes.txt` (doporučeno nasamplovat ručně).

- **Závislosti**: Python 3.12+, knihovny: `soundfile`, `numpy`, `resampy`, `tqdm`, `re`, `pathlib`, `argparse`, `sys`, `collections`.

## Instalace

1. Nainstalujte Python 3.12+.
2. Nainstalujte závislosti příkazem:
   ```
   pip install soundfile numpy resampy tqdm
   ```
   (Ostatní knihovny jsou standardní.)

## Použití

Spusťte skript s argumenty:

```
python sample_gap_filler.py --input-dir CESTA_K_VSTUPU --output-dir CESTA_K_VYSTUPU [--verbose]
```

- `--input-dir`: Cesta k adresáři s existujícími vzorky (výstup z Pitch Corrector).
- `--output-dir`: Cesta k výstupnímu adresáři (bude vytvořen, pokud neexistuje).
- `--verbose`: Volitelný, zapne podrobný výstup (vypne progress bary).

**Příklad**:
```
python sample_gap_filler.py --input-dir ./processed_samples --output-dir ./complete_samples --verbose
```

## Poznámky

- Program používá jednoduchý pitch shift (změna sample rate), který mění délku vzorku – vhodné pro samplery.
- Pokud nelze najít zdrojový vzorek v rozsahu ±3 půltónů, vzorek se nedoplní a zapíše do reportu.

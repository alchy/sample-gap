# Simple Sample Gap Filler

Jednoduchý program pro automatické doplňování chybějících vzorků hudebních nástrojů transpozicí z nejbližších dostupných vzorků. Vybírá vždy **NEJBLIŽŠÍ** dostupnou notu bez ohledu na směr.

## 🎯 Účel

Když máte neúplnou sadu vzorků hudebního nástroje, tento nástroj automaticky vygeneruje chybějící noty transpozicí z nejbližších existujících vzorků. Ideální pro:

- Doplnění chybějících not v sample knihovnách
- Rozšíření MIDI rozsahu nástrojů
- Příprava kompletních sad pro IthacaSampler
- Rychlé a jednoduché zpracování vzorků

## ✨ Co je nového ve verzi 3.0

### 🔄 **Kompletní přepis**
- **Jednodušší kód** - odstranění zbytečných složitostí
- **Čistší logika** - méně tříd, jasnější flow
- **Lepší error handling** - opraveny syntaxe chyby

### 🎵 **Zjednodušený výběr zdroje**
- **Nejbližší dostupná nota** - prostě podle nejmenší vzdálenosti
- **Žádné složité priority** - dolů/nahoru logika odstraněna
- **Rychlejší hledání** - méně složitých podmínek

### 🛠️ **Pouze librosa**
- **Povinná librosa** - odstraněny problematické alternativy
- **Konzistentní kvalita** - vše přes jednu knihovnu
- **Žádné konflikty** - končí s duplikáty bez změn

## 📋 Požadavky

### Povinná závislost
```bash
pip install librosa soundfile numpy tqdm
```

**Poznámka:** librosa je nyní povinná a program se odmítne spustit bez ní.

## 🚀 Instalace a použití

### 1. Základní použití
```bash
python sample-gap.py --input-dir ./processed_samples --output-dir ./complete_samples
```

### 2. Bez kopírování originálů
```bash
python sample-gap.py --input-dir ./processed_samples --output-dir ./complete_samples --no-copy
```

### 3. Verbose režim pro debugging
```bash
python sample-gap.py --input-dir ./processed_samples --output-dir ./complete_samples --verbose
```

### 4. Příklad pro Windows
```bash
python sample-gap.py --input-dir "C:\SoundBanks\IthacaPlayer\PianoP-tuned" --output-dir "C:\SoundBanks\IthacaPlayer\instrument"
```

## 📁 Konvence názvů souborů

Program očekává soubory ve formátu:
```
m{midi:03d}-vel{velocity}-f{sample_rate}[-next{N}].wav
```

### Příklady:
- `m060-vel3-f44.wav` - C4, velocity 3, 44.1kHz
- `m072-vel5-f48.wav` - C5, velocity 5, 48kHz  
- `m036-vel1-f44-next1.wav` - C2, velocity 1, 44.1kHz, duplicita 1

### Parametry:
- **midi**: 000-127 (MIDI číslo noty)
- **velocity**: 0-7 (velocity layer)
- **sample_rate**: 44 (=44100Hz) nebo 48 (=48000Hz)
- **next**: Volitelný index duplicity (1, 2, 3...)

## ⚙️ Parametry příkazové řádky

| Parametr | Popis |
|----------|-------|
| `--input-dir` | Cesta k adresáři s existujícími vzorky (povinné) |
| `--output-dir` | Cesta k výstupnímu adresáři (povinné) |
| `--no-copy` | Nekopírovat originální vzorky |
| `--verbose` | Podrobný výstup pro debugging |

## 🔄 Jak to funguje

### **1. Skenování vzorků**
- Načte všechny WAV soubory podle konvence názvů
- Ignoruje soubory s neplatným formátem názvu
- Zobrazí statistiky načtených vzorků

### **2. Analýza pokrytí**
- Vypočte požadované vzorky (21-108 MIDI × 0-7 velocity × 2 sample rates)
- Identifikuje chybějící kombinace
- Zobrazí přehled co je potřeba vygenerovat

### **3. Generování vzorků**
- Pro každý chybějící vzorek najde **nejbližší dostupný zdroj**
- Aplikuje pitch shift pro dosažení cílové noty
- Konvertuje sample rate pokud je potřeba
- Uloží vygenerovaný vzorek

### **4. Kopírování originálů**
- Zkopíruje všechny originální vzorky do výstupního adresáře
- Přeskočí již existující soubory
- Volitelné (lze vypnout pomocí `--no-copy`)

## 🎛️ Logika výběru zdroje

### **Jednoduchý algoritmus**
```python
# Najde vzorek s nejmenší vzdáleností od cíle
distance = abs(source_midi - target_midi)
if distance <= MAX_TRANSPOSE:  # max ±3 půltóny
    use_this_source()
```

### **Priority při hledání**
1. **Stejný velocity + sample rate** s nejmenší vzdáleností
2. **Jiný velocity + stejný sample rate** s nejmenší vzdáleností  
3. **Jakýkoliv** s nejmenší vzdáleností

### **Bez složitých pravidel**
- ❌ Žádné "dolů má prioritu před nahoru"
- ❌ Žádné složité fallbacky
- ✅ Prostě nejbližší dostupná nota

## 📊 Výstupní formáty

### **Generované vzorky**
- Zachován originální formát (mono/stereo)
- Stejná délka jako zdrojový vzorek
- Cílový sample rate (44.1kHz nebo 48kHz)
- Vysoká kvalita zpracování přes librosa

### **Report soubory**
```
missing-samples.txt    # Seznam vzorků, které nebylo možné vygenerovat
```

## 🔧 Konfigurační konstanty

```python
MIDI_RANGE = (21, 108)        # A0 - C8
VELOCITY_RANGE = (0, 7)       # Velocity layers
SAMPLE_RATES = [44100, 48000] # Podporované sample rates
MAX_TRANSPOSE = 3             # ±3 půltóny max transpozice
```

### **Úprava MAX_TRANSPOSE**
Pokud chcete vygenerovat více vzorků, můžete zvýšit maximální vzdálenost transpozice:

```python
# V souboru sample-gap.py, řádek cca 80
self.MAX_TRANSPOSE = 5  # Změnit z 3 na 5 pro ±5 půltónů
```

**Doporučení:**
- **3 půltóny** - nejlepší kvalita, méně artefaktů
- **4-5 půltónů** - více vygenerovaných vzorků, mírně nižší kvalita
- **6+ půltónů** - může způsobit výrazné artefakty v sound

## 📈 Příklad workflow

```bash
# 1. Zpracování vzorků Pitch Correctorem (předchozí krok)
python pitch_corrector.py --input-dir ./raw_samples --output-dir ./processed_samples

# 2. Doplnění chybějících vzorků (NOVÁ VERZE)
python sample-gap.py --input-dir ./processed_samples --output-dir ./complete_samples

# 3. Kontrola výsledků
ls -la ./complete_samples/
cat ./complete_samples/missing-samples.txt
```

## 🐛 Troubleshooting

### **Chyba: librosa není k dispozici**
```bash
pip install librosa
```

### **Verbose režim**
Při problémech použijte `--verbose` pro detailní výstup:
```bash
python sample-gap.py --input-dir ./input --output-dir ./output --verbose
```

### **Časté problémy:**

**Nerozpoznané názvy souborů**
- Zkontrolujte formát názvů: `m{midi:03d}-vel{velocity}-f{sr}.wav`
- MIDI musí být 3 cifry: `m060`, ne `m60`

**Syntaxe chyba**
- Používejte uvozovky pro cesty s mezerami
- Windows: `"C:\Path With Spaces\folder"`

**Nedostatečné zdroje (576 vzorků selhalo)**
- Zkontrolujte `missing-samples.txt` pro seznam nedostupných vzorků
- **Zvýšte MAX_TRANSPOSE** z 3 na 4-5 pro více vygenerovaných vzorků
- Vysoké noty (jako C8) jsou těžce dostupné - zvažte ruční nasamplování

## ⚡ Výhody verze 3.0

### **Rychlost**
- ✅ Jednodušší algoritmus = rychlejší zpracování
- ✅ Méně podmínek při hledání zdroje
- ✅ Přímočařejší logika

### **Spolehlivost**
- ✅ Pouze librosa = konzistentní kvalita
- ✅ Žádné konflikty mezi knihovnami
- ✅ Opravené syntaxe chyby

### **Jednoduchost**
- ✅ Méně parametrů = méně možností pro chyby
- ✅ Čistší kód = snadnější údržba
- ✅ Jasná logika = předvídatelné výsledky

## 🔗 Související nástroje

- **Pitch Corrector** - Předchozí krok v pipeline
- **IthacaSampler** - Cílová aplikace pro vzorky

## 📝 Changelog

### **Verze 3.0 (Aktuální)**
- ✅ **KOMPLETNÍ PŘEPIS** - zjednodušená architektura
- ✅ **Nejbližší nota** - odstraněna složitá logika směru
- ✅ **Pouze librosa** - žádné problematické alternativy
- ✅ **Opravené syntaxe** - odstraněna syntax error
- ✅ **Čistší API** - `--no-copy` místo `--do-not-copy-source`

"""
Program pro doplňování chybějících vzorků hudebních nástrojů
transpozicí z nejbližších dostupných vzorků.

Navazuje na Pitch Corrector a používá stejné principy zpracování.

Autor: Doplňkový program pro IthacaSampler
Datum: 2025
"""

import argparse
import soundfile as sf
import numpy as np
import resampy
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import re
import sys


class ProgressManager:
    """Správce progress barů a výstupu"""

    def __init__(self, verbose=False):
        self.verbose = verbose

    def info(self, message):
        """Informační zpráva"""
        if self.verbose:
            print(f"[INFO] {message}")
        else:
            tqdm.write(f"[INFO] {message}")

    def debug(self, message):
        """Debug zpráva (pouze v verbose režimu)"""
        if self.verbose:
            print(f"[DEBUG] {message}")

    def warning(self, message):
        """Varovná zpráva"""
        if self.verbose:
            print(f"[WARNING] {message}")
        else:
            tqdm.write(f"[WARNING] {message}")

    def error(self, message):
        """Chybová zpráva"""
        if self.verbose:
            print(f"[ERROR] {message}")
        else:
            tqdm.write(f"[ERROR] {message}")

    def section(self, title):
        """Sekce - hlavička"""
        separator = "=" * len(title)
        if self.verbose:
            print(f"\n{separator}")
            print(title)
            print(separator)
        else:
            tqdm.write(f"\n{separator}")
            tqdm.write(title)
            tqdm.write(separator)


class AudioUtils:
    """Pomocné funkce pro práci s audio daty - sdílené s Pitch Corrector"""

    MIDI_TO_NOTE = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }

    @staticmethod
    def midi_to_freq(midi):
        """Převod MIDI čísla na frekvenci"""
        return 440.0 * 2 ** ((midi - 69) / 12)

    @staticmethod
    def midi_to_note_name(midi):
        """Převod MIDI čísla na název noty (např. 60 -> C4)"""
        if not (0 <= midi <= 127):
            raise ValueError(f"MIDI číslo {midi} je mimo povolený rozsah 0-127")

        octave = (midi // 12) - 1
        note_idx = midi % 12
        note = AudioUtils.MIDI_TO_NOTE[note_idx]
        if len(note) == 1:
            note += '_'
        return f"{note}{octave}"

    @staticmethod
    def normalize_audio(waveform, target_db=-20.0):
        """Normalizace audio na cílovou úroveň v dB"""
        if len(waveform.shape) > 1:
            rms = np.sqrt(np.mean(waveform.flatten() ** 2))
        else:
            rms = np.sqrt(np.mean(waveform ** 2))

        if rms == 0:
            return waveform

        target_rms = 10 ** (target_db / 20)
        return waveform * (target_rms / rms)


class SimplePitchShifter:
    """Jednoduchý pitch shifting pomocí resample - stejný jako v Pitch Corrector"""

    def __init__(self, progress_mgr=None):
        self.progress_mgr = progress_mgr or ProgressManager()

    def pitch_shift_simple(self, audio, sr, semitone_shift):
        """
        Pitch shift změnou sample rate - mění délku vzorku.
        Stejná implementace jako v Pitch Corrector.
        """
        if abs(semitone_shift) < 0.01:
            return audio, sr

        factor = 2 ** (semitone_shift / 12)
        new_sr = int(sr * factor)

        try:
            # Zpracování multi-channel audio
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                shifted_channels = []
                for ch in range(audio.shape[1]):
                    shifted = resampy.resample(audio[:, ch], sr, new_sr)
                    shifted_channels.append(shifted)
                shifted_audio = np.column_stack(shifted_channels)
            else:
                # Mono nebo 2D s jedním kanálem
                audio_1d = audio.flatten() if len(audio.shape) > 1 else audio
                shifted_audio = resampy.resample(audio_1d, sr, new_sr)

                # Zachovat originální formát
                if len(audio.shape) > 1:
                    shifted_audio = shifted_audio[:, np.newaxis]

            return shifted_audio, new_sr

        except Exception as e:
            self.progress_mgr.error(f"Chyba při pitch shift: {e}")
            return audio, sr


class SampleInfo:
    """Container pro informace o vzorku"""

    def __init__(self, filepath, midi, velocity, sample_rate):
        self.filepath = Path(filepath)
        self.midi = midi
        self.velocity = velocity
        self.sample_rate = sample_rate
        self.note_name = AudioUtils.midi_to_note_name(midi)

    def __str__(self):
        return f"MIDI {self.midi} ({self.note_name}) vel{self.velocity} @ {self.sample_rate}Hz"


class SampleGapFiller:
    """
    Hlavní třída pro doplňování chybějících vzorků transpozicí z nejbližších.
    """

    def __init__(self, input_dir, output_dir, verbose=False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # Konstanty
        self.MIDI_MIN = 21  # A0
        self.MIDI_MAX = 108  # C8
        self.VELOCITY_MIN = 0
        self.VELOCITY_MAX = 7
        self.MAX_TRANSPOSE_DISTANCE = 3  # ±3 půltóny
        self.TARGET_SAMPLE_RATES = [44100, 48000]

        # Validace adresářů
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Vstupní adresář neexistuje: {self.input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Inicializace komponent
        self.progress_mgr = ProgressManager(verbose=verbose)
        self.pitch_shifter = SimplePitchShifter(progress_mgr=self.progress_mgr)

        # Regex pro parsování názvů souborů
        # Očekává formát: m{midi:03d}-vel{velocity}-f{sample_rate}.wav
        self.filename_pattern = re.compile(r'm(\d{3})-vel(\d+)-f(\d+)\.wav$', re.IGNORECASE)

    def parse_filename(self, filepath):
        """Parsování informací z názvu souboru"""
        match = self.filename_pattern.match(filepath.name)
        if not match:
            return None

        try:
            midi = int(match.group(1))
            velocity = int(match.group(2))
            sample_rate = int(match.group(3))

            # Validace rozsahů
            if not (0 <= midi <= 127):
                return None
            if not (0 <= velocity <= 7):
                return None
            if sample_rate not in [44100, 48000]:
                return None

            return SampleInfo(filepath, midi, velocity, sample_rate)

        except (ValueError, IndexError):
            return None

    def scan_existing_samples(self):
        """Fáze 1: Skenování existujících vzorků"""
        self.progress_mgr.section("FÁZE 1: Skenování existujících vzorků")

        wav_files = list(self.input_dir.glob("*.wav")) + list(self.input_dir.glob("*.WAV"))

        if not wav_files:
            self.progress_mgr.error("Nebyly nalezeny žádné WAV soubory!")
            return {}

        self.progress_mgr.info(f"Nalezeno {len(wav_files)} WAV souborů")

        # Index: (midi, velocity, sample_rate) -> SampleInfo
        existing_samples = {}

        iterator = wav_files if self.verbose else tqdm(wav_files, desc="Skenuji soubory", unit="soubor")

        for filepath in iterator:
            sample_info = self.parse_filename(filepath)

            if sample_info is None:
                self.progress_mgr.warning(f"Nerozpoznaný formát názvu: {filepath.name}")
                continue

            key = (sample_info.midi, sample_info.velocity, sample_info.sample_rate)
            existing_samples[key] = sample_info

            self.progress_mgr.debug(f"Indexován: {sample_info}")

        self.progress_mgr.info(f"Úspěšně indexováno {len(existing_samples)} vzorků")
        return existing_samples

    def analyze_coverage(self, existing_samples):
        """Fáze 2: Analýza pokrytí MIDI rozsahu"""
        self.progress_mgr.section("FÁZE 2: Analýza pokrytí")

        # Vytvoření kompletního seznamu požadovaných vzorků
        required_samples = set()
        for midi in range(self.MIDI_MIN, self.MIDI_MAX + 1):
            for velocity in range(self.VELOCITY_MIN, self.VELOCITY_MAX + 1):
                for sample_rate in self.TARGET_SAMPLE_RATES:
                    required_samples.add((midi, velocity, sample_rate))

        existing_keys = set(existing_samples.keys())
        missing_samples = required_samples - existing_keys

        self.progress_mgr.info(f"Celkem požadováno: {len(required_samples)} vzorků")
        self.progress_mgr.info(f"Existuje: {len(existing_keys)} vzorků")
        self.progress_mgr.info(f"Chybí: {len(missing_samples)} vzorků")

        # Statistiky podle velocity layers
        if self.verbose:
            for velocity in range(self.VELOCITY_MIN, self.VELOCITY_MAX + 1):
                existing_for_vel = sum(1 for (m, v, sr) in existing_keys if v == velocity)
                missing_for_vel = sum(1 for (m, v, sr) in missing_samples if v == velocity)
                total_for_vel = existing_for_vel + missing_for_vel
                coverage = (existing_for_vel / total_for_vel * 100) if total_for_vel > 0 else 0
                print(f"  Velocity {velocity}: {existing_for_vel}/{total_for_vel} ({coverage:.1f}%)")

        return missing_samples

    def find_source_sample(self, target_midi, target_velocity, target_sr, existing_samples):
        """
        Nalezení nejbližšího existujícího vzorku pro transpozici.
        Priorita: stejný velocity, nejbližší MIDI (dolů má prioritu), stejný sample rate.
        """
        # Nejdřív hledáme ve stejném sample rate
        candidates = []

        # Procházíme vzdálenosti: -1, -2, -3, +1, +2, +3
        for distance in range(1, self.MAX_TRANSPOSE_DISTANCE + 1):
            # Nejdřív dolů (priorita)
            source_midi = target_midi - distance
            key = (source_midi, target_velocity, target_sr)
            if key in existing_samples:
                return existing_samples[key], -distance

            # Pak nahoru
            source_midi = target_midi + distance
            key = (source_midi, target_velocity, target_sr)
            if key in existing_samples:
                return existing_samples[key], distance

        # Pokud nenajdeme ve stejném sample rate, zkusíme druhý
        other_sr = 48000 if target_sr == 44100 else 44100
        for distance in range(1, self.MAX_TRANSPOSE_DISTANCE + 1):
            # Nejdřív dolů
            source_midi = target_midi - distance
            key = (source_midi, target_velocity, other_sr)
            if key in existing_samples:
                return existing_samples[key], -distance

            # Pak nahoru
            source_midi = target_midi + distance
            key = (source_midi, target_velocity, other_sr)
            if key in existing_samples:
                return existing_samples[key], distance

        return None, 0

    def generate_missing_sample(self, source_sample, semitone_shift, target_midi, target_velocity, target_sr):
        """Generování chybějícího vzorku transpozicí ze zdroje"""
        try:
            # Načtení zdrojového audio
            waveform, source_sr = sf.read(str(source_sample.filepath))

            # Zajištění 2D formátu
            if len(waveform.shape) == 1:
                waveform = waveform[:, np.newaxis]

            self.progress_mgr.debug(f"Načten zdroj: {waveform.shape[0]} vzorků, {source_sr}Hz")

            # Pitch shift
            shifted_waveform, shifted_sr = self.pitch_shifter.pitch_shift_simple(
                waveform, source_sr, semitone_shift
            )

            # Konverze na cílový sample rate pokud potřeba
            if shifted_sr != target_sr:
                if len(shifted_waveform.shape) > 1 and shifted_waveform.shape[1] > 1:
                    # Multi-channel
                    converted_channels = []
                    for ch in range(shifted_waveform.shape[1]):
                        converted = resampy.resample(shifted_waveform[:, ch], shifted_sr, target_sr)
                        converted_channels.append(converted)
                    output_waveform = np.column_stack(converted_channels)
                else:
                    # Mono
                    waveform_flat = shifted_waveform.flatten() if len(shifted_waveform.shape) > 1 else shifted_waveform
                    output_waveform = resampy.resample(waveform_flat, shifted_sr, target_sr)
                    output_waveform = output_waveform[:, np.newaxis]
            else:
                output_waveform = shifted_waveform

            return output_waveform, target_sr

        except Exception as e:
            self.progress_mgr.error(f"Chyba při generování vzorku: {e}")
            return None, None

    def generate_filename(self, midi, velocity, sample_rate):
        """Generování názvu souboru podle konvence"""
        sr_suffix = 'f44' if sample_rate == 44100 else 'f48'
        return f"m{midi:03d}-vel{velocity}-{sr_suffix}.wav"

    def fill_gaps(self, missing_samples, existing_samples):
        """Fáze 3: Doplňování chybějících vzorků"""
        self.progress_mgr.section("FÁZE 3: Generování chybějících vzorků")

        if not missing_samples:
            self.progress_mgr.info("Žádné vzorky k doplnění!")
            return 0, []

        generated_count = 0
        truly_missing = []

        # Seřazení pro lepší progress tracking
        missing_list = sorted(list(missing_samples))

        iterator = missing_list if self.verbose else tqdm(missing_list, desc="Generuji vzorky", unit="vzorek")

        for midi, velocity, sample_rate in iterator:
            note_name = AudioUtils.midi_to_note_name(midi)
            filename = self.generate_filename(midi, velocity, sample_rate)

            if self.verbose:
                print(f"\n--- Generuji: {filename} ---")
            else:
                tqdm.write(f"\nGeneruji: {filename}")

            # Hledání zdrojového vzorku
            source_sample, semitone_distance = self.find_source_sample(
                midi, velocity, sample_rate, existing_samples
            )

            if source_sample is None:
                self.progress_mgr.warning(f"Nelze najít zdroj pro {filename}")
                truly_missing.append(filename)
                continue

            # Výpočet transpozice
            semitone_shift = semitone_distance  # Vzdálenost už je se správným znaménkem

            info_lines = [
                f"  Zdroj: {source_sample.filepath.name}",
                f"  Transpozice: {semitone_shift:+d} půltónů"
            ]

            for line in info_lines:
                if self.verbose:
                    print(line)
                else:
                    tqdm.write(line)

            # Generování vzorku
            output_waveform, output_sr = self.generate_missing_sample(
                source_sample, semitone_shift, midi, velocity, sample_rate
            )

            if output_waveform is None:
                self.progress_mgr.error(f"Selhala generace pro {filename}")
                truly_missing.append(filename)
                continue

            # Uložení
            output_path = self.output_dir / filename
            try:
                sf.write(str(output_path), output_waveform, output_sr)

                save_info = f"  Uložen: {filename}"
                if self.verbose:
                    print(save_info)
                else:
                    tqdm.write(save_info)

                generated_count += 1

            except Exception as e:
                self.progress_mgr.error(f"Chyba při ukládání {filename}: {e}")
                truly_missing.append(filename)

        return generated_count, truly_missing

    def save_missing_report(self, missing_samples):
        """Uložení reportu o vzorcích, které se nepodařilo vygenerovat"""
        if not missing_samples:
            return

        report_path = self.output_dir / "missing-notes.txt"

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# Vzorky které nebylo možné vygenerovat\n")
                f.write("# Doporučuje se nasamplovat ručně\n\n")

                for filename in sorted(missing_samples):
                    f.write(f"{filename}\n")

            self.progress_mgr.warning(f"Report chybějících vzorků uložen: {report_path}")

        except Exception as e:
            self.progress_mgr.error(f"Chyba při ukládání reportu: {e}")

    def process_all(self):
        """Hlavní pipeline zpracování"""
        print(f"Vstupní adresář: {self.input_dir}")
        print(f"Výstupní adresář: {self.output_dir}")
        print(f"MIDI rozsah: {self.MIDI_MIN}-{self.MIDI_MAX}")
        print(f"Velocity layers: {self.VELOCITY_MIN}-{self.VELOCITY_MAX}")
        print(f"Max transpozice: ±{self.MAX_TRANSPOSE_DISTANCE} půltóny")
        print(f"Cílové sample rates: {self.TARGET_SAMPLE_RATES}")
        print(f"Verbose režim: {'ZAPNUT' if self.verbose else 'VYPNUT'}")

        try:
            # Fáze 1: Skenování existujících vzorků
            existing_samples = self.scan_existing_samples()
            if not existing_samples:
                return

            # Fáze 2: Analýza pokrytí
            missing_samples = self.analyze_coverage(existing_samples)

            # Fáze 3: Doplňování chybějících vzorků
            generated_count, truly_missing = self.fill_gaps(missing_samples, existing_samples)

            # Fáze 4: Report nedostupných vzorků
            if truly_missing:
                self.save_missing_report(truly_missing)

            # Finální shrnutí
            self.progress_mgr.section("DOKONČENO")
            summary_lines = [
                f"Vygenerováno: {generated_count} vzorků",
                f"Nepodařilo se: {len(truly_missing)} vzorků",
                f"Výstupní adresář: {self.output_dir}"
            ]

            if truly_missing:
                summary_lines.append(f"Viz report: missing-notes.txt")

            for line in summary_lines:
                if self.verbose:
                    print(line)
                else:
                    tqdm.write(line)

        except KeyboardInterrupt:
            self.progress_mgr.error("Zpracování přerušeno uživatelem")
        except Exception as e:
            self.progress_mgr.error(f"Neočekávaná chyba: {e}")
            raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Program pro doplňování chybějících vzorků transpozicí z nejbližších dostupných.

        Klíčové funkce:
        - Indexování existujících vzorků podle názvu (m{midi:03d}-vel{velocity}-f{sr}.wav)
        - Analýza pokrytí MIDI rozsahu 21-108 (A0-C8) pro velocity layers 0-7
        - Generování chybějících vzorků transpozicí z nejbližších (±3 půltóny max)
        - Priorita směru transpozice: dolů (-1, -2, -3), pak nahoru (+1, +2, +3)
        - Export pro oba sample rates (44.1kHz + 48kHz)
        - Report nedostupných vzorků do missing-notes.txt

        Navazuje na Pitch Corrector a používá stejné metody transpozice.

        Příklad použití:
        python sample_gap_filler.py --input-dir ./processed_samples --output-dir ./complete_samples --verbose
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--input-dir', required=True,
                        help='Cesta k adresáři s existujícími vzorky (výstup z Pitch Corrector)')
    parser.add_argument('--output-dir', required=True,
                        help='Cesta k výstupnímu adresáři pro doplněné vzorky')
    parser.add_argument('--verbose', action='store_true',
                        help='Podrobný výstup pro debugging (vypne progress bary)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=== SAMPLE GAP FILLER ===")
    print("Doplňování chybějících vzorků pro IthacaSampler")
    print("Transpozice z nejbližších dostupných vzorků")
    print("=" * 50)

    try:
        filler = SampleGapFiller(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            verbose=args.verbose
        )

        filler.process_all()

    except FileNotFoundError as e:
        print(f"CHYBA: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nZpracování přerušeno uživatelem")
        sys.exit(1)
    except Exception as e:
        print(f"KRITICKÁ CHYBA: {e}")
        sys.exit(1)
"""
Program pro doplňování chybějících vzorků hudebních nástrojů
transpozicí z nejbližších dostupných vzorků s quality-aware algoritmem.

Navazuje na Pitch Corrector a používá stejné principy zpracování.

Autor: Doplňkový program pro IthacaSampler s kvalitní analýzou
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
import shutil

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

    def __init__(self, filepath, midi, velocity, sample_rate, is_duplicate=False, duplicate_index=0):
        self.filepath = Path(filepath)
        self.midi = midi
        self.velocity = velocity
        self.sample_rate = sample_rate
        self.note_name = AudioUtils.midi_to_note_name(midi)
        self.is_duplicate = is_duplicate
        self.duplicate_index = duplicate_index  # 0 = originál, 1 = -next1, atd.

    def __str__(self):
        dup_info = f" (dup-{self.duplicate_index})" if self.is_duplicate else ""
        return f"MIDI {self.midi} ({self.note_name}) vel{self.velocity} @ {self.sample_rate}Hz{dup_info}"


class QualityAnalyzer:
    """Analýza kvality vzorků na základě počtu verzí každé MIDI noty"""

    def __init__(self, progress_mgr=None):
        self.progress_mgr = progress_mgr or ProgressManager()

    def analyze_note_quality(self, samples_by_midi, quality_threshold=0.8):
        """
        Analýza kvality not na základě počtu vzorků.
        Více vzorků = lepší kvalita (uživatel častěji samploval)

        Args:
            samples_by_midi: Dict {midi: [SampleInfo, ...]}
            quality_threshold: Paretovo pravidlo (0.8 = vyřadí spodních 20%)

        Returns:
            (good_midi_notes, excluded_midi_notes, quality_stats)
        """
        self.progress_mgr.section("KVALITNÍ ANALÝZA VZORKŮ")

        # Spočítej vzorky pro každou MIDI notu (včetně duplicitů)
        midi_sample_counts = {}
        for midi, samples in samples_by_midi.items():
            total_samples = len(samples)  # Včetně duplicitů (-next1, -next2, atd.)
            midi_sample_counts[midi] = total_samples

        if not midi_sample_counts:
            return set(), set(), {}

        # Výpočet mediánu a statistik
        sample_counts = list(midi_sample_counts.values())
        median_samples = np.median(sample_counts)
        min_samples = min(sample_counts)
        max_samples = max(sample_counts)

        self.progress_mgr.info(f"Statistiky vzorků na notu:")
        self.progress_mgr.info(f"  Min: {min_samples}, Medián: {median_samples:.1f}, Max: {max_samples}")

        # Seřazení not podle kvality (více vzorků = lepší)
        sorted_notes = sorted(midi_sample_counts.items(), key=lambda x: x[1], reverse=True)

        # Paretovo pravidlo - vyřaď spodní (1 - quality_threshold) procent
        total_notes = len(sorted_notes)
        keep_count = int(total_notes * quality_threshold)
        exclude_count = total_notes - keep_count

        good_notes = set(midi for midi, _ in sorted_notes[:keep_count])
        excluded_notes = set(midi for midi, _ in sorted_notes[keep_count:])

        self.progress_mgr.info(f"Paretovo pravidlo ({quality_threshold*100:.0f}%):")
        self.progress_mgr.info(f"  Kvalitní noty: {len(good_notes)} (zachováno)")
        self.progress_mgr.info(f"  Vyloučené noty: {len(excluded_notes)} (nepoužijí se pro klonování)")

        # Detailní statistiky pro verbose režim
        if self.progress_mgr.verbose:
            print(f"\nVyloučené noty (méně vzorků = horší kvalita):")
            for midi in sorted(excluded_notes):
                count = midi_sample_counts[midi]
                note_name = AudioUtils.midi_to_note_name(midi)
                print(f"  MIDI {midi} ({note_name}): {count} vzorků")

        quality_stats = {
            'median': median_samples,
            'min': min_samples,
            'max': max_samples,
            'total_notes': total_notes,
            'good_count': len(good_notes),
            'excluded_count': len(excluded_notes)
        }

        return good_notes, excluded_notes, quality_stats


class SampleGapFiller:
    """
    Hlavní třída pro doplňování chybějících vzorků transpozicí z nejbližších
    s quality-aware algoritmem.
    """

    def __init__(self, input_dir, output_dir, quality_threshold=0.8, verbose=False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.quality_threshold = quality_threshold
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
        self.quality_analyzer = QualityAnalyzer(progress_mgr=self.progress_mgr)

        # Regex pro parsování názvů souborů
        # Formát: m{midi:03d}-vel{velocity}-f{44|48}[-next{N}].wav
        self.filename_pattern = re.compile(
            r'm(\d{3})-vel(\d+)-f(44|48)(?:-next(\d+))?\.wav$',
            re.IGNORECASE
        )

    def parse_filename(self, filepath):
        """Parsování informací z názvu souboru včetně duplicitů"""
        match = self.filename_pattern.match(filepath.name)
        if not match:
            return None

        try:
            midi = int(match.group(1))
            velocity = int(match.group(2))
            sr_str = match.group(3)
            next_index_str = match.group(4)  # None pokud není -next

            # Mapování zkráceného formátu na plnou hodnotu sample rate
            if sr_str == '44':
                sample_rate = 44100
            elif sr_str == '48':
                sample_rate = 48000
            else:
                return None

            # Validace rozsahů
            if not (0 <= midi <= 127):
                return None
            if not (0 <= velocity <= 7):
                return None
            if sample_rate not in [44100, 48000]:
                return None

            # Detekce duplicitu
            is_duplicate = next_index_str is not None
            duplicate_index = int(next_index_str) if next_index_str else 0

            return SampleInfo(filepath, midi, velocity, sample_rate, is_duplicate, duplicate_index)

        except (ValueError, IndexError):
            return None

    def scan_existing_samples(self):
        """Fáze 1: Skenování existujících vzorků s podporou duplicitů"""
        self.progress_mgr.section("FÁZE 1: Skenování existujících vzorků")

        wav_files = list(self.input_dir.glob("*.wav")) + list(self.input_dir.glob("*.WAV"))

        if not wav_files:
            self.progress_mgr.error("Nebyly nalezeny žádné WAV soubory!")
            return {}, {}

        self.progress_mgr.info(f"Nalezeno {len(wav_files)} WAV souborů")

        # Index: (midi, velocity, sample_rate) -> SampleInfo
        existing_samples = {}
        # Index pro kvalitní analýzu: midi -> [SampleInfo, ...]
        samples_by_midi = defaultdict(list)

        iterator = wav_files if self.verbose else tqdm(wav_files, desc="Skenuji soubory", unit="soubor")

        duplicates_found = 0

        for filepath in iterator:
            sample_info = self.parse_filename(filepath)

            if sample_info is None:
                self.progress_mgr.warning(f"Nerozpoznaný formát názvu: {filepath.name}")
                continue

            # Přidej do hlavního indexu
            key = (sample_info.midi, sample_info.velocity, sample_info.sample_rate)
            existing_samples[key] = sample_info

            # Přidej do MIDI indexu pro kvalitní analýzu
            samples_by_midi[sample_info.midi].append(sample_info)

            if sample_info.is_duplicate:
                duplicates_found += 1

            self.progress_mgr.debug(f"Indexován: {sample_info}")

        self.progress_mgr.info(f"Úspěšně indexováno {len(existing_samples)} vzorků")
        self.progress_mgr.info(f"Z toho duplicitů (-next): {duplicates_found}")
        self.progress_mgr.info(f"Unikátních MIDI not: {len(samples_by_midi)}")

        return existing_samples, dict(samples_by_midi)

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

    def find_source_sample(self, target_midi, target_velocity, target_sr, existing_samples, good_midi_notes):
        """
        Nalezení nejbližšího existujícího vzorku pro transpozici.
        Priorita: stejný velocity, nejbližší MIDI (dolů má prioritu), stejný sample rate.
        POUZE z kvalitních not (good_midi_notes).
        """
        # Nejdřív hledáme ve stejném sample rate
        for distance in range(1, self.MAX_TRANSPOSE_DISTANCE + 1):
            # Nejdřív dolů (priorita)
            source_midi = target_midi - distance
            if source_midi in good_midi_notes:  # KONTROLA KVALITY
                key = (source_midi, target_velocity, target_sr)
                if key in existing_samples:
                    return existing_samples[key], -distance

            # Pak nahoru
            source_midi = target_midi + distance
            if source_midi in good_midi_notes:  # KONTROLA KVALITY
                key = (source_midi, target_velocity, target_sr)
                if key in existing_samples:
                    return existing_samples[key], distance

        # Pokud nenajdeme ve stejném sample rate, zkusíme druhý
        other_sr = 48000 if target_sr == 44100 else 44100
        for distance in range(1, self.MAX_TRANSPOSE_DISTANCE + 1):
            # Nejdřív dolů
            source_midi = target_midi - distance
            if source_midi in good_midi_notes:  # KONTROLA KVALITY
                key = (source_midi, target_velocity, other_sr)
                if key in existing_samples:
                    return existing_samples[key], -distance

            # Pak nahoru
            source_midi = target_midi + distance
            if source_midi in good_midi_notes:  # KONTROLA KVALITY
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

    def fill_gaps(self, missing_samples, existing_samples, good_midi_notes, excluded_midi_notes):
        """Fáze 3: Doplňování chybějících vzorků (pouze z kvalitních zdrojů)"""
        self.progress_mgr.section("FÁZE 3: Generování chybějících vzorků")

        if not missing_samples:
            self.progress_mgr.info("Žádné vzorky k doplnění!")
            return 0, []

        generated_count = 0
        truly_missing = []
        excluded_due_to_quality = []

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

            # Hledání zdrojového vzorku POUZE z kvalitních not
            source_sample, semitone_distance = self.find_source_sample(
                midi, velocity, sample_rate, existing_samples, good_midi_notes
            )

            if source_sample is None:
                # Zkontroluj, zda by se našel zdroj bez omezení kvality
                source_any, _ = self.find_source_sample_any_quality(
                    midi, velocity, sample_rate, existing_samples
                )

                if source_any is not None:
                    self.progress_mgr.warning(f"Vyloučeno kvůli kvalitě: {filename}")
                    excluded_due_to_quality.append(filename)
                else:
                    self.progress_mgr.warning(f"Nelze najít žádný zdroj pro: {filename}")
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

        return generated_count, truly_missing, excluded_due_to_quality

    def find_source_sample_any_quality(self, target_midi, target_velocity, target_sr, existing_samples):
        """Pomocná metoda pro kontrolu dostupnosti zdrojů bez omezení kvality"""
        for distance in range(1, self.MAX_TRANSPOSE_DISTANCE + 1):
            # Dolů
            source_midi = target_midi - distance
            key = (source_midi, target_velocity, target_sr)
            if key in existing_samples:
                return existing_samples[key], -distance

            # Nahoru
            source_midi = target_midi + distance
            key = (source_midi, target_velocity, target_sr)
            if key in existing_samples:
                return existing_samples[key], distance

        # Druhý sample rate
        other_sr = 48000 if target_sr == 44100 else 44100
        for distance in range(1, self.MAX_TRANSPOSE_DISTANCE + 1):
            source_midi = target_midi - distance
            key = (source_midi, target_velocity, other_sr)
            if key in existing_samples:
                return existing_samples[key], -distance

            source_midi = target_midi + distance
            key = (source_midi, target_velocity, other_sr)
            if key in existing_samples:
                return existing_samples[key], distance

        return None, 0

    def copy_quality_originals(self, existing_samples, good_midi_notes, excluded_midi_notes):
        """Fáze 4: Kopírování kvalitních originálních vzorků"""
        self.progress_mgr.section("FÁZE 4: Kopírování originálů")

        copied_count = 0
        excluded_count = 0

        # Filtrování vzorků podle kvality
        good_samples = []
        for sample_info in existing_samples.values():
            if sample_info.midi in good_midi_notes:
                good_samples.append(sample_info)
            else:
                excluded_count += 1

        if not good_samples:
            self.progress_mgr.warning("Žádné kvalitní vzorky ke kopírování!")
            return 0, excluded_count

        self.progress_mgr.info(f"Kopíruji {len(good_samples)} kvalitních originálů")
        self.progress_mgr.info(f"Vylučuji {excluded_count} vzorků kvůli nízké kvalitě")

        iterator = good_samples if self.verbose else tqdm(good_samples, desc="Kopíruji originály", unit="soubor")

        for sample_info in iterator:
            source_path = sample_info.filepath
            dest_path = self.output_dir / source_path.name

            try:
                # Přeskoč pokud již existuje (může být vygenerovaný)
                if dest_path.exists():
                    self.progress_mgr.debug(f"Přeskakuji existující: {dest_path.name}")
                    continue

                shutil.copy2(source_path, dest_path)
                copied_count += 1

                if self.verbose:
                    print(f"Zkopírován: {source_path.name}")

            except Exception as e:
                self.progress_mgr.error(f"Chyba při kopírování {source_path.name}: {e}")

        return copied_count, excluded_count

    def save_missing_report(self, missing_samples, excluded_due_to_quality=None):
        """Uložení reportu o vzorcích, které se nepodařilo vygenerovat"""
        if not missing_samples and not excluded_due_to_quality:
            return

        report_path = self.output_dir / "missing-notes.txt"

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# Report chybějících vzorků\n")
                f.write("# Vygenerováno Sample Gap Filler s quality-aware algoritmem\n\n")

                if missing_samples:
                    f.write("## Vzorky které nebylo možné vygenerovat\n")
                    f.write("# Doporučuje se nasamplovat ručně\n\n")
                    for filename in sorted(missing_samples):
                        f.write(f"{filename}\n")

                if excluded_due_to_quality:
                    f.write("\n## Vzorky vyloučené kvůli nízké kvalitě zdrojů\n")
                    f.write("# Mohly by se vygenerovat, ale pouze z nekvalitních not\n\n")
                    for filename in sorted(excluded_due_to_quality):
                        f.write(f"{filename}\n")

            self.progress_mgr.warning(f"Report uložen: {report_path}")

        except Exception as e:
            self.progress_mgr.error(f"Chyba při ukládání reportu: {e}")

    def process_all(self):
        """Hlavní pipeline zpracování s quality-aware algoritmem"""
        print(f"Vstupní adresář: {self.input_dir}")
        print(f"Výstupní adresář: {self.output_dir}")
        print(f"MIDI rozsah: {self.MIDI_MIN}-{self.MIDI_MAX}")
        print(f"Velocity layers: {self.VELOCITY_MIN}-{self.VELOCITY_MAX}")
        print(f"Max transpozice: ±{self.MAX_TRANSPOSE_DISTANCE} půltóny")
        print(f"Quality threshold: {self.quality_threshold*100:.0f}% (Paretovo pravidlo)")
        print(f"Cílové sample rates: {self.TARGET_SAMPLE_RATES}")
        print(f"Verbose režim: {'ZAPNUT' if self.verbose else 'VYPNUT'}")

        try:
            # Fáze 1: Skenování existujících vzorků
            existing_samples, samples_by_midi = self.scan_existing_samples()
            if not existing_samples:
                return

            # Fáze 1.5: Analýza kvality vzorků
            good_midi_notes, excluded_midi_notes, quality_stats = self.quality_analyzer.analyze_note_quality(
                samples_by_midi, self.quality_threshold
            )

            # Fáze 2: Analýza pokrytí
            missing_samples = self.analyze_coverage(existing_samples)

            # Fáze 3: Doplňování chybějících vzorků (pouze z kvalitních zdrojů)
            generated_count, truly_missing, excluded_due_to_quality = self.fill_gaps(
                missing_samples, existing_samples, good_midi_notes, excluded_midi_notes
            )

            # Fáze 4: Kopírování kvalitních originálů
            copied_count, excluded_originals = self.copy_quality_originals(
                existing_samples, good_midi_notes, excluded_midi_notes
            )

            # Fáze 5: Report nedostupných vzorků
            if truly_missing or excluded_due_to_quality:
                self.save_missing_report(truly_missing, excluded_due_to_quality)

            # Finální shrnutí
            self.progress_mgr.section("DOKONČENO")
            summary_lines = [
                f"Kvalitní analýza:",
                f"  • Medián vzorků na notu: {quality_stats.get('median', 0):.1f}",
                f"  • Kvalitních not: {quality_stats.get('good_count', 0)}",
                f"  • Vyloučených not: {quality_stats.get('excluded_count', 0)}",
                f"",
                f"Výsledky:",
                f"  • Vygenerováno: {generated_count} vzorků",
                f"  • Zkopírováno originálů: {copied_count} vzorků",
                f"  • Vyloučeno kvůli kvalitě: {len(excluded_due_to_quality)} vzorků",
                f"  • Nelze vygenerovat: {len(truly_missing)} vzorků",
                f"",
                f"Výstupní adresář: {self.output_dir}"
            ]

            if truly_missing or excluded_due_to_quality:
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
        description="""Program pro doplňování chybějících vzorků transpozicí z nejbližších dostupných
s quality-aware algoritmem.

        Klíčové funkce:
        - Indexování existujících vzorků podle názvu (m{midi:03d}-vel{velocity}-f{sr}[-next{N}].wav)
        - Quality-aware analýza: více vzorků = lepší kvalita (častěji samplované noty)
        - Paretovo pravidlo pro vyřazení nejhorších not z transpozice
        - Analýza pokrytí MIDI rozsahu 21-108 (A0-C8) pro velocity layers 0-7
        - Generování chybějících vzorků transpozicí POUZE z kvalitních zdrojů
        - Priorita směru transpozice: dolů (-1, -2, -3), pak nahoru (+1, +2, +3)
        - Kopírování kvalitních originálů + export pro oba sample rates (44.1kHz + 48kHz)
        - Detailní report vyloučených vzorků do missing-notes.txt

        Navazuje na Pitch Corrector a používá stejné metody transpozice.

        Příklad použití:
        python sample_gap_filler.py --input-dir ./processed_samples --output-dir ./complete_samples --quality-threshold 0.8 --verbose
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--input-dir', required=True,
                        help='Cesta k adresáři s existujícími vzorky (výstup z Pitch Corrector)')
    parser.add_argument('--output-dir', required=True,
                        help='Cesta k výstupnímu adresáři pro doplněné vzorky')
    parser.add_argument('--quality-threshold', type=float, default=0.8,
                        help='Paretovo pravidlo pro kvalitu (0.8 = ponechá top 80%%, vyřadí spodních 20%% not)')
    parser.add_argument('--verbose', action='store_true',
                        help='Podrobný výstup pro debugging (vypne progress bary)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=== SAMPLE GAP FILLER s QUALITY-AWARE ALGORITMEM ===")
    print("Doplňování chybějících vzorků pro IthacaSampler")
    print("Transpozice pouze z kvalitních vzorků (Paretovo pravidlo)")
    print("=" * 60)

    # Validace quality threshold
    if not (0.1 <= args.quality_threshold <= 1.0):
        print("CHYBA: Quality threshold musí být mezi 0.1 a 1.0")
        sys.exit(1)

    try:
        filler = SampleGapFiller(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            quality_threshold=args.quality_threshold,
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
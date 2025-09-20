#!/usr/bin/env python3
"""
Simple Sample Gap Filler v3.0 - Kompletně přepsaný

Jednoduchý program pro doplňování chybějících vzorků transpozicí.
Vybírá NEJBLIŽŠÍ dostupnou notu bez ohledu na směr.

Autor: Simple Sample Gap Filler pro IthacaSampler
Datum: 2025
Verze: 3.0 (KOMPLETNÍ PŘEPIS)
"""

import argparse
import soundfile as sf
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import re
import sys
import shutil

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class AudioProcessor:
    """Jednoduchá třída pro audio zpracování - pouze librosa"""
    
    @staticmethod
    def pitch_shift(audio, sr, semitones):
        """Pitch shift pomocí librosa"""
        if abs(semitones) < 0.01:
            return audio
            
        if len(audio.shape) > 1:
            # Multi-channel
            channels = []
            for ch in range(audio.shape[1]):
                shifted = librosa.effects.pitch_shift(audio[:, ch], sr=sr, n_steps=semitones)
                channels.append(shifted)
            return np.column_stack(channels)
        else:
            # Mono
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
    
    @staticmethod
    def resample(audio, orig_sr, target_sr):
        """Sample rate konverze pomocí librosa"""
        if orig_sr == target_sr:
            return audio
            
        if len(audio.shape) > 1:
            # Multi-channel
            channels = []
            for ch in range(audio.shape[1]):
                resampled = librosa.resample(audio[:, ch], orig_sr=orig_sr, target_sr=target_sr)
                channels.append(resampled)
            return np.column_stack(channels)
        else:
            # Mono
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


class Sample:
    """Jednoduchá třída pro reprezentaci vzorku"""
    
    def __init__(self, filepath, midi, velocity, sample_rate):
        self.filepath = Path(filepath)
        self.midi = midi
        self.velocity = velocity
        self.sample_rate = sample_rate
        self.note_name = self._midi_to_note(midi)
    
    def _midi_to_note(self, midi):
        """Převod MIDI na název noty"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi // 12) - 1
        note = notes[midi % 12]
        return f"{note}{octave}"
    
    def __str__(self):
        return f"MIDI {self.midi} ({self.note_name}) vel{self.velocity} @ {self.sample_rate}Hz"


class GapFiller:
    """Hlavní třída pro doplňování vzorků"""
    
    def __init__(self, input_dir, output_dir, copy_originals=True, verbose=False):
        # Kontrola librosa
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa je povinná! Nainstalujte: pip install librosa")
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.should_copy_originals = copy_originals  # Přejmenováno kvůli konfliktu s metodou
        self.verbose = verbose
        
        # Konstanty
        self.MIDI_RANGE = (21, 108)  # A0 - C8
        self.VELOCITY_RANGE = (0, 7)
        self.SAMPLE_RATES = [44100, 48000]
        self.MAX_TRANSPOSE = 3  # max ±3 půltóny
        
        # Validace
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Vstupní adresář neexistuje: {self.input_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data
        self.samples = {}  # (midi, velocity, sr) -> [Sample, ...]
        
        # Regex pro parsování
        self.filename_pattern = re.compile(r'm(\d{3})-vel(\d+)-f(44|48)(?:-next(\d+))?\.wav$', re.IGNORECASE)
    
    def log(self, message, level="INFO"):
        """Jednoduché logování"""
        if self.verbose:
            print(f"[{level}] {message}")
        elif level in ["WARNING", "ERROR"]:
            tqdm.write(f"[{level}] {message}")
    
    def parse_filename(self, filepath):
        """Parsování názvu souboru"""
        match = self.filename_pattern.match(filepath.name)
        if not match:
            return None
        
        try:
            midi = int(match.group(1))
            velocity = int(match.group(2))
            sr_code = match.group(3)
            
            # Validace
            if not (0 <= midi <= 127):
                return None
            if not (0 <= velocity <= 7):
                return None
            
            sample_rate = 44100 if sr_code == '44' else 48000
            
            return Sample(filepath, midi, velocity, sample_rate)
            
        except (ValueError, IndexError):
            return None
    
    def scan_samples(self):
        """Skenování existujících vzorků"""
        print("Skenování vzorků...")
        
        wav_files = list(self.input_dir.glob("*.wav")) + list(self.input_dir.glob("*.WAV"))
        
        if not wav_files:
            raise FileNotFoundError("Nebyly nalezeny žádné WAV soubory!")
        
        valid_count = 0
        
        for filepath in tqdm(wav_files, desc="Načítání", disable=self.verbose):
            sample = self.parse_filename(filepath)
            if sample:
                key = (sample.midi, sample.velocity, sample.sample_rate)
                if key not in self.samples:
                    self.samples[key] = []
                self.samples[key].append(sample)
                valid_count += 1
                self.log(f"Načten: {sample}")
            else:
                self.log(f"Neplatný název: {filepath.name}", "WARNING")
        
        print(f"Načteno {valid_count} platných vzorků z {len(wav_files)} souborů")
        print(f"Unikátních kombinací: {len(self.samples)}")
    
    def find_missing(self):
        """Najití chybějících vzorků"""
        print("Analýza chybějících vzorků...")
        
        required = set()
        for midi in range(self.MIDI_RANGE[0], self.MIDI_RANGE[1] + 1):
            for velocity in range(self.VELOCITY_RANGE[0], self.VELOCITY_RANGE[1] + 1):
                for sr in self.SAMPLE_RATES:
                    required.add((midi, velocity, sr))
        
        existing = set(self.samples.keys())
        missing = required - existing
        
        print(f"Požadováno: {len(required)}")
        print(f"Existuje: {len(existing)}")
        print(f"Chybí: {len(missing)}")
        
        return sorted(missing)
    
    def find_closest_source(self, target_midi, target_velocity, target_sr):
        """
        Najde NEJBLIŽŠÍ dostupný vzorek pro transpozici.
        Jednoduše podle vzdálenosti - bez ohledu na směr.
        """
        best_sample = None
        best_distance = float('inf')
        
        # Projdi všechny dostupné vzorky
        for (midi, velocity, sr), samples in self.samples.items():
            # Zkus stejný velocity a sample rate
            if velocity == target_velocity and sr == target_sr:
                distance = abs(midi - target_midi)
                if distance <= self.MAX_TRANSPOSE and distance < best_distance:
                    best_distance = distance
                    best_sample = random.choice(samples)
        
        # Pokud nenašel ve stejném velocity, zkus jiné
        if best_sample is None:
            for (midi, velocity, sr), samples in self.samples.items():
                if sr == target_sr:  # alespoň stejný sample rate
                    distance = abs(midi - target_midi)
                    if distance <= self.MAX_TRANSPOSE and distance < best_distance:
                        best_distance = distance
                        best_sample = random.choice(samples)
        
        # Pokud nenašel ani s jiným sample rate, zkus úplně všechno
        if best_sample is None:
            for (midi, velocity, sr), samples in self.samples.items():
                distance = abs(midi - target_midi)
                if distance <= self.MAX_TRANSPOSE and distance < best_distance:
                    best_distance = distance
                    best_sample = random.choice(samples)
        
        if best_sample:
            semitone_shift = target_midi - best_sample.midi
            return best_sample, semitone_shift
        
        return None, 0
    
    def generate_sample(self, source_sample, semitone_shift, target_sr):
        """Generování vzorku transpozicí"""
        try:
            # Načtení
            audio, orig_sr = sf.read(str(source_sample.filepath))
            
            # Zajisti 2D formát
            if len(audio.shape) == 1:
                audio = audio[:, np.newaxis]
            
            self.log(f"Načten: {audio.shape}, {orig_sr}Hz")
            
            # Pitch shift
            if abs(semitone_shift) >= 0.01:
                self.log(f"Pitch shift: {semitone_shift:+.1f} semitónů")
                audio = AudioProcessor.pitch_shift(audio, orig_sr, semitone_shift)
            
            # Sample rate konverze
            if orig_sr != target_sr:
                self.log(f"Resample: {orig_sr}Hz -> {target_sr}Hz")
                audio = AudioProcessor.resample(audio, orig_sr, target_sr)
            
            return audio, target_sr
            
        except Exception as e:
            self.log(f"Chyba při generování: {e}", "ERROR")
            return None, None
    
    def generate_filename(self, midi, velocity, sr):
        """Generování názvu souboru"""
        sr_code = '44' if sr == 44100 else '48'
        return f"m{midi:03d}-vel{velocity}-f{sr_code}.wav"
    
    def fill_gaps(self):
        """Hlavní funkce pro doplnění mezer"""
        missing = self.find_missing()
        
        if not missing:
            print("Žádné vzorky k doplnění!")
            return
        
        print(f"Generování {len(missing)} chybějících vzorků...")
        
        generated = 0
        failed = []
        
        for midi, velocity, sr in tqdm(missing, desc="Generování", disable=self.verbose):
            filename = self.generate_filename(midi, velocity, sr)
            
            if self.verbose:
                print(f"\n--- {filename} ---")
            
            # Najdi zdroj
            source, shift = self.find_closest_source(midi, velocity, sr)
            
            if source is None:
                self.log(f"Nelze najít zdroj pro {filename}", "WARNING")
                failed.append(filename)
                continue
            
            self.log(f"Zdroj: {source.filepath.name} (shift: {shift:+d})")
            
            # Generuj
            audio, out_sr = self.generate_sample(source, shift, sr)
            
            if audio is None:
                failed.append(filename)
                continue
            
            # Ulož
            output_path = self.output_dir / filename
            try:
                sf.write(str(output_path), audio, out_sr)
                generated += 1
                self.log(f"Uložen: {filename}")
            except Exception as e:
                self.log(f"Chyba při ukládání {filename}: {e}", "ERROR")
                failed.append(filename)
        
        print(f"Vygenerováno: {generated}")
        print(f"Selhalo: {len(failed)}")
        
        if failed:
            self._save_failed_report(failed)
    
    def copy_originals(self):
        """Kopírování originálních souborů"""
        if not self.should_copy_originals:
            print("Kopírování originálů přeskočeno")
            return
        
        print("Kopírování originálních vzorků...")
        
        all_samples = []
        for samples_list in self.samples.values():
            all_samples.extend(samples_list)
        
        copied = 0
        
        for sample in tqdm(all_samples, desc="Kopírování", disable=self.verbose):
            dest = self.output_dir / sample.filepath.name
            
            if dest.exists():
                self.log(f"Existuje: {sample.filepath.name}")
                continue
            
            try:
                shutil.copy2(sample.filepath, dest)
                copied += 1
                self.log(f"Zkopírován: {sample.filepath.name}")
            except Exception as e:
                self.log(f"Chyba kopírování {sample.filepath.name}: {e}", "ERROR")
        
        print(f"Zkopírováno: {copied} originálů")
    
    def _save_failed_report(self, failed_files):
        """Uložení reportu neúspěšných vzorků"""
        report_path = self.output_dir / "missing-samples.txt"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# Vzorky, které se nepodařilo vygenerovat\n\n")
                for filename in sorted(failed_files):
                    f.write(f"{filename}\n")
            
            print(f"Report uložen: {report_path}")
            
        except Exception as e:
            self.log(f"Chyba při ukládání reportu: {e}", "ERROR")
    
    def process(self):
        """Hlavní zpracování"""
        print("=== SIMPLE SAMPLE GAP FILLER v3.0 ===")
        print(f"Vstup: {self.input_dir}")
        print(f"Výstup: {self.output_dir}")
        print(f"MIDI rozsah: {self.MIDI_RANGE[0]}-{self.MIDI_RANGE[1]}")
        print(f"Velocity: {self.VELOCITY_RANGE[0]}-{self.VELOCITY_RANGE[1]}")
        print(f"Sample rates: {self.SAMPLE_RATES}")
        print(f"Max transpozice: ±{self.MAX_TRANSPOSE} půltónů")
        print(f"Kopírovat originály: {'ANO' if self.should_copy_originals else 'NE'}")
        print("=" * 50)
        
        try:
            self.scan_samples()
            self.fill_gaps()
            self.copy_originals()
            
            print("\n=== DOKONČENO ===")
            
        except Exception as e:
            print(f"CHYBA: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="""
Simple Sample Gap Filler v3.0 - Kompletně přepsaný

Jednoduchý program pro doplňování chybějících vzorků transpozicí.
Vždy vybere NEJBLIŽŠÍ dostupnou notu bez ohledu na směr.

Vyžaduje librosa: pip install librosa

Formát názvů: m{midi:03d}-vel{velocity}-f{sr}[-next{N}].wav
Příklad: m060-vel3-f44.wav (C4, velocity 3, 44.1kHz)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input-dir', required=True,
                       help='Adresář s existujícími vzorky')
    parser.add_argument('--output-dir', required=True,
                       help='Výstupní adresář')
    parser.add_argument('--no-copy', action='store_true',
                       help='Nekopírovat originální vzorky')
    parser.add_argument('--verbose', action='store_true',
                       help='Podrobný výstup')
    
    args = parser.parse_args()
    
    try:
        filler = GapFiller(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            copy_originals=not args.no_copy,
            verbose=args.verbose
        )
        
        filler.process()
        
    except ImportError as e:
        print(f"CHYBA ZÁVISLOSTI: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"CHYBA: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPřerušeno uživatelem")
        sys.exit(1)
    except Exception as e:
        print(f"NEOČEKÁVANÁ CHYBA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
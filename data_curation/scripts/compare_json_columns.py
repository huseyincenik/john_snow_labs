#!/usr/bin/env python3
"""
JSON dosyalarının sütun yapılarını karşılaştıran script.
Extractor, tagger ve consolidator JSON dosyaları için ayrı ayrı analiz yapar.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any
from datetime import datetime


def extract_all_keys(obj: Any, prefix: str = "", max_depth: int = 10) -> Set[str]:
    """
    JSON objesindeki tüm anahtarları recursive olarak çıkarır.

    Args:
        obj: JSON objesi (dict, list, veya primitive)
        prefix: Anahtar öneki (nested yapılar için)
        max_depth: Maksimum derinlik (sonsuz döngüyü önlemek için)

    Returns:
        Tüm anahtarların set'i
    """
    keys = set()

    if max_depth <= 0:
        return keys

    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.add(full_key)

            # Nested yapıları da işle
            if isinstance(value, (dict, list)):
                nested_keys = extract_all_keys(value, full_key, max_depth - 1)
                keys.update(nested_keys)

    elif isinstance(obj, list) and len(obj) > 0:
        # Listenin ilk elemanını kullan (tüm elemanların aynı yapıda olduğunu varsay)
        nested_keys = extract_all_keys(obj[0], prefix, max_depth - 1)
        keys.update(nested_keys)

    return keys


def get_structure_summary(obj: Any, prefix: str = "", max_depth: int = 5) -> Dict[str, Any]:
    """
    JSON objesinin yapı özetini oluşturur (anahtarlar ve veri tipleri).
    """
    summary = {}

    if max_depth <= 0:
        return summary

    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                summary[full_key] = "dict"
                nested = get_structure_summary(value, full_key, max_depth - 1)
                summary.update(nested)
            elif isinstance(value, list):
                summary[full_key] = f"list[{len(value)}]"
                if len(value) > 0:
                    nested = get_structure_summary(value[0], full_key, max_depth - 1)
                    summary.update(nested)
            else:
                summary[full_key] = type(value).__name__

    elif isinstance(obj, list) and len(obj) > 0:
        nested = get_structure_summary(obj[0], prefix, max_depth - 1)
        summary.update(nested)

    return summary


def categorize_file(filename: str) -> str:
    """
    Dosya adına göre dosya türünü belirler.
    """
    filename_lower = filename.lower()
    if "extractor" in filename_lower:
        return "extractor"
    elif "tagger" in filename_lower:
        return "tagger"
    elif "consolidator" in filename_lower:
        return "consolidator"
    else:
        return "unknown"


def compare_files(file_paths: List[Path], category: str) -> Dict[str, Any]:
    """
    Aynı kategorideki dosyaları karşılaştırır.
    """
    results = {
        "category": category,
        "file_count": len(file_paths),
        "files": [],
        "all_keys": set(),
        "common_keys": set(),
        "unique_keys_per_file": {},
        "structure_comparison": {},
    }

    all_keys_sets = []
    file_keys_map = {}

    # Her dosyayı oku ve anahtarları çıkar
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            keys = extract_all_keys(data)
            all_keys_sets.append(keys)
            file_keys_map[file_path.name] = keys
            results["all_keys"].update(keys)

            # Dosya bilgilerini kaydet
            file_info = {
                "filename": file_path.name,
                "total_keys": len(keys),
                "keys": sorted(keys),
                "structure": get_structure_summary(data),
            }
            results["files"].append(file_info)

        except Exception as e:
            print(f"Hata: {file_path.name} okunurken hata oluştu: {e}")
            continue

    # Ortak anahtarları bul
    if all_keys_sets:
        results["common_keys"] = set.intersection(*all_keys_sets)

        # Her dosya için benzersiz anahtarları bul
        for filename, keys in file_keys_map.items():
            other_keys = set()
            for other_filename, other_keys_set in file_keys_map.items():
                if other_filename != filename:
                    other_keys.update(other_keys_set)

            unique_keys = keys - other_keys
            results["unique_keys_per_file"][filename] = sorted(unique_keys)

    return results


def generate_report(results: Dict[str, Dict[str, Any]], output_dir: Path) -> str:
    """
    Detaylı bir rapor oluşturur ve dosyaya yazar.
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("JSON DOSYALARI SÜTUN KARŞILAŞTIRMA RAPORU")
    report_lines.append("=" * 80)
    report_lines.append(f"Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    for category, category_results in results.items():
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append(f"KATEGORİ: {category.upper()}")
        report_lines.append("=" * 80)
        report_lines.append("")

        report_lines.append(f"Toplam Dosya Sayısı: {category_results['file_count']}")
        report_lines.append("")

        # Dosya listesi
        report_lines.append("Dosyalar:")
        for i, file_info in enumerate(category_results["files"], 1):
            file_type = "SAMPLE (output_samples)" if i == 1 else "OUTPUT (data/output)"
            report_lines.append(f"  {i}. [{file_type}] {file_info['filename']}")
            report_lines.append(f"     Toplam Anahtar Sayısı: {file_info['total_keys']}")
        report_lines.append("")

        # Ortak anahtarlar
        common_keys = category_results["common_keys"]
        report_lines.append(f"Ortak Anahtarlar ({len(common_keys)} adet):")
        if common_keys:
            for key in sorted(common_keys):
                report_lines.append(f"  - {key}")
        else:
            report_lines.append("  (Ortak anahtar bulunamadı)")
        report_lines.append("")

        # Tüm anahtarlar
        all_keys = category_results["all_keys"]
        report_lines.append(f"Tüm Anahtarlar ({len(all_keys)} adet):")
        for key in sorted(all_keys):
            report_lines.append(f"  - {key}")
        report_lines.append("")

        # Benzersiz anahtarlar
        unique_keys = category_results["unique_keys_per_file"]
        if unique_keys and category_results["file_count"] == 2:
            report_lines.append("BENZERSİZ ANAHTARLAR:")
            report_lines.append("")

            # Sample ve Output dosyalarını ayır
            sample_file_info = (
                category_results["files"][0] if len(category_results["files"]) > 0 else None
            )
            output_file_info = (
                category_results["files"][1] if len(category_results["files"]) > 1 else None
            )

            if sample_file_info and output_file_info:
                sample_keys = set(sample_file_info["keys"])
                output_keys = set(output_file_info["keys"])

                only_in_sample = sample_keys - output_keys
                only_in_output = output_keys - sample_keys

                if only_in_sample:
                    report_lines.append(
                        f"⚠ SADECE SAMPLE'DA OLAN ANAHTARLAR ({len(only_in_sample)} adet):"
                    )
                    report_lines.append(f"   (Bu anahtarlar Output dosyasında YOK)")
                    for key in sorted(only_in_sample):
                        report_lines.append(f"   - {key}")
                    report_lines.append("")

                if only_in_output:
                    report_lines.append(
                        f"⚠ SADECE OUTPUT'TA OLAN ANAHTARLAR ({len(only_in_output)} adet):"
                    )
                    report_lines.append(f"   (Bu anahtarlar Sample dosyasında YOK)")
                    for key in sorted(only_in_output):
                        report_lines.append(f"   - {key}")
                    report_lines.append("")

                if not only_in_sample and not only_in_output:
                    report_lines.append(
                        "✓ Tüm anahtarlar her iki dosyada da mevcut (Benzersiz anahtar yok)"
                    )
                    report_lines.append("")

            # Eski formatı da göster (dosya adına göre)
            report_lines.append("Dosyalara Göre Benzersiz Anahtarlar:")
            for filename, keys in unique_keys.items():
                if keys:
                    report_lines.append(f"  {filename}:")
                    for key in keys:
                        report_lines.append(f"    - {key}")
                else:
                    report_lines.append(f"  {filename}: (Benzersiz anahtar yok)")
            report_lines.append("")

        # Yapı karşılaştırması
        if category_results["file_count"] > 1:
            report_lines.append("Yapı Tutarlılığı:")
            all_consistent = True
            first_file_keys = set(category_results["files"][0]["keys"])

            for i, file_info in enumerate(category_results["files"][1:], 1):
                current_keys = set(file_info["keys"])
                if first_file_keys != current_keys:
                    all_consistent = False
                    report_lines.append(
                        f"  ⚠️  Dosya {i+1} ({file_info['filename']}) yapısı farklı!"
                    )
                    missing = first_file_keys - current_keys
                    extra = current_keys - first_file_keys
                    if missing:
                        report_lines.append(f"     Eksik anahtarlar: {len(missing)} adet")
                    if extra:
                        report_lines.append(f"     Fazla anahtarlar: {len(extra)} adet")
                else:
                    report_lines.append(f"  ✓  Dosya {i+1} ({file_info['filename']}) yapısı aynı")

            if all_consistent:
                report_lines.append("  ✓  Tüm dosyalar aynı yapıya sahip!")
            report_lines.append("")

    # Özet
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("ÖZET")
    report_lines.append("=" * 80)
    report_lines.append("")

    for category, category_results in results.items():
        report_lines.append(f"{category.upper()}:")
        report_lines.append(f"  - Dosya Sayısı: {category_results['file_count']}")
        report_lines.append(f"  - Toplam Benzersiz Anahtar: {len(category_results['all_keys'])}")
        report_lines.append(f"  - Ortak Anahtar: {len(category_results['common_keys'])}")

        if category_results["file_count"] > 1:
            consistency = (
                "✓ Tutarlı"
                if len(category_results["common_keys"]) == len(category_results["all_keys"])
                else "⚠ Tutarsız"
            )
            report_lines.append(f"  - Tutarlılık: {consistency}")
        report_lines.append("")

    report_text = "\n".join(report_lines)

    # Raporu dosyaya yaz
    report_file = (
        output_dir / f"json_column_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    return str(report_file)


def main():
    """
    Ana fonksiyon
    output_samples klasöründeki dosyalar ile data/output altındaki belirli dosyaları karşılaştırır.
    """
    # Klasörleri bul
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_samples_dir = project_root / "output_samples"
    data_output_dir = project_root / "data" / "output" / "469c14d1-ea30-4b88-802f-2c96f6b3c46a"

    if not output_samples_dir.exists():
        print(f"Hata: {output_samples_dir} klasörü bulunamadı!")
        return

    if not data_output_dir.exists():
        print(f"Hata: {data_output_dir} klasörü bulunamadı!")
        return

    # Karşılaştırılacak dosya çiftlerini tanımla
    file_pairs = {
        "tagger": {
            "sample": "stage_tagger_48a759f7-2a79-4b98-86ff-dd8a4094e97b_sorted.json",
            "output": "stage_tagger_469c14d1-ea30-4b88-802f-2c96f6b3c46a_sorted.json",
        },
        "extractor": {
            "sample": "stage_extractor_48a759f7-2a79-4b98-86ff-dd8a4094e97b_extraction.json",
            "output": "stage_extractor_469c14d1-ea30-4b88-802f-2c96f6b3c46a_extraction.json",
        },
        "consolidator": {
            "sample": "stage_consolidator_48a759f7-2a79-4b98-86ff-dd8a4094e97b_consolidation.json",
            "output": "stage_consolidator_469c14d1-ea30-4b88-802f-2c96f6b3c46a_consolidation.json",
        },
    }

    # Her kategori için karşılaştırma yap
    all_results = {}

    for category, file_names in file_pairs.items():
        print(f"\n{category.upper()} kategorisi analiz ediliyor...")

        sample_file = output_samples_dir / file_names["sample"]
        output_file = data_output_dir / file_names["output"]

        # Dosyaların varlığını kontrol et
        files_to_compare = []

        if sample_file.exists():
            files_to_compare.append(sample_file)
            print(f"  ✓ Sample dosyası bulundu: {sample_file.name}")
        else:
            print(f"  ⚠ Sample dosyası bulunamadı: {sample_file.name}")

        if output_file.exists():
            files_to_compare.append(output_file)
            print(f"  ✓ Output dosyası bulundu: {output_file.name}")
        else:
            print(f"  ⚠ Output dosyası bulunamadı: {output_file.name}")

        if len(files_to_compare) < 2:
            print(f"  ⚠ Uyarı: {category} için yeterli dosya bulunamadı, atlanıyor.")
            continue

        # Karşılaştırma yap
        results = compare_files(files_to_compare, category)
        all_results[category] = results

        print(
            f"  - Sample anahtar sayısı: {len(results['files'][0]['keys']) if len(results['files']) > 0 else 0}"
        )
        print(
            f"  - Output anahtar sayısı: {len(results['files'][1]['keys']) if len(results['files']) > 1 else 0}"
        )
        print(f"  - Ortak anahtar: {len(results['common_keys'])}")

        # Benzersiz anahtarları göster
        unique_keys = results["unique_keys_per_file"]
        for filename, keys in unique_keys.items():
            if keys:
                print(f"  - {filename} benzersiz anahtarlar: {len(keys)} adet")

    if not all_results:
        print("\nHata: Hiçbir kategori için karşılaştırma yapılamadı!")
        return

    # Rapor oluştur
    print("\nRapor oluşturuluyor...")
    report_file = generate_report(all_results, output_samples_dir)
    print(f"\n✓ Rapor oluşturuldu: {report_file}")

    # Konsola da özet yazdır
    print("\n" + "=" * 80)
    print("ÖZET")
    print("=" * 80)
    for category, results in all_results.items():
        print(f"\n{category.upper()}:")
        print(f"  Dosya Sayısı: {results['file_count']}")
        print(f"  Toplam Anahtar: {len(results['all_keys'])}")
        print(f"  Ortak Anahtar: {len(results['common_keys'])}")

        if results["file_count"] == 2:
            sample_keys = set(results["files"][0]["keys"]) if len(results["files"]) > 0 else set()
            output_keys = set(results["files"][1]["keys"]) if len(results["files"]) > 1 else set()

            only_in_sample = sample_keys - output_keys
            only_in_output = output_keys - sample_keys

            if only_in_sample:
                print(f"  ⚠ Sample'da olup Output'ta olmayan: {len(only_in_sample)} adet")
            if only_in_output:
                print(f"  ⚠ Output'ta olup Sample'da olmayan: {len(only_in_output)} adet")

            if len(results["common_keys"]) == len(results["all_keys"]):
                print(f"  Durum: ✓ Tüm anahtarlar aynı (Mükemmel uyumluluk)")
            elif len(only_in_sample) == 0:
                print(f"  Durum: ✓ İyi (Output'ta ekstra anahtarlar var)")
            elif len(only_in_output) == 0:
                print(f"  Durum: ✓ İyi (Sample'da ekstra anahtarlar var)")
            else:
                print(f"  Durum: ⚠ Uyumsuzluk var (Her iki tarafta da benzersiz anahtarlar var)")


if __name__ == "__main__":
    main()

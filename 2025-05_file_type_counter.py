#!/usr/bin/env python3
import os
import argparse
import zipfile
import sys
import time
from collections import defaultdict

try:
    import magic
except ImportError:
    print("Please install python-magic (e.g. pip install python-magic)")
    sys.exit(1)

# Metadata definitions for known file extensions:
# Each key is an extension and maps to:
#   - description: human-readable description
#   - mime_types: typical MIME types
#   - processing: notes on how to process these files
EXTENSION_METADATA = {
    '7z': {
        'description': '7-Zip compressed archive',
        'mime_types': ['application/x-7z-compressed'],
        'processing': 'do not inspect contents by default'
    },
    'bmpr': {
        'description': 'Balsamiq Mockups project file',
        'mime_types': ['application/xml'],
        'processing': 'Treated as XML; could extract UI elements if needed'
    },
    'zip': {
        'description': 'ZIP archive',
        'mime_types': ['application/zip'],
        'processing': 'skip deep inspection'
    },
    'ctrl': {
        'description': 'Control file (custom enterprise definition)',
        'mime_types': ['text/plain'],
        'processing': 'categorize by extension only'
    },
    'dat': {
        'description': 'Generic data file',
        'mime_types': ['application/octet-stream'],
        'processing': 'unknown internal format'
    },
    'drawio': {
        'description': 'draw.io diagram (ZIP-backed)',
        'mime_types': ['application/zip'],
        'processing': 'Detected via ZIP check; could unzip and inspect XML'
    },
    'no_extension': {
        'description': 'Files with no extension',
        'mime_types': ['various'],
        'processing': 'Use MIME detection; classify further if needed'
    },
    'ics': {
        'description': 'iCalendar file',
        'mime_types': ['text/calendar'],
        'processing': 'Could parse events; currently only counted'
    },
    'java': {
        'description': 'Java source code',
        'mime_types': ['text/x-java-source'],
        'processing': 'could run static analysis'
    },
    'jil': {
        'description': 'Autosys job definition',
        'mime_types': ['text/plain'],
        'processing': 'parse JIL syntax if needed'
    },
    'jpb': {
        'description': 'Jobplan backup (custom)',
        'mime_types': ['application/octet-stream'],
        'processing': 'proprietary format'
    },
    'md': {
        'description': 'Markdown document',
        'mime_types': ['text/markdown'],
        'processing': 'could render to HTML'
    },
    'pdf': {
        'description': 'PDF document',
        'mime_types': ['application/pdf'],
        'processing': 'could extract text/images'
    },
    'xls': {
        'description': 'Excel 97-2003 workbook',
        'mime_types': ['application/vnd.ms-excel'],
        'processing': 'could read via xlrd'
    },
    'xlsb': {
        'description': 'Excel binary workbook',
        'mime_types': ['application/vnd.ms-excel.sheet.binary.macroenabled.12'],
        'processing': 'could read via pyxlsb'
    },
    'csv': {
        'description': 'Comma-separated values',
        'mime_types': ['text/csv'],
        'processing': 'parse as needed'
    },
    'xlsx': {
        'description': 'Excel Open XML workbook',
        'mime_types': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
        'processing': 'could read via openpyxl'
    },
    'pptx': {
        'description': 'PowerPoint Open XML presentation',
        'mime_types': ['application/vnd.openxmlformats-officedocument.presentationml.presentation'],
        'processing': 'could parse slides'
    },
    'ppsx': {
        'description': 'PowerPoint Open XML slideshow',
        'mime_types': ['application/vnd.openxmlformats-officedocument.presentationml.slideshow'],
        'processing': 'could parse slides'
    },
    'sql': {
        'description': 'SQL script',
        'mime_types': ['application/sql'],
        'processing': 'could perform syntax checks'
    },
    'docx': {
        'description': 'Word Open XML document',
        'mime_types': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
        'processing': 'could extract text via python-docx'
    },
    'msg': {
        'description': 'Outlook message file',
        'mime_types': ['application/vnd.ms-outlook'],
        'processing': 'could parse via extract_msg'
    },
    'png': {
        'description': 'PNG image',
        'mime_types': ['image/png'],
        'processing': 'could generate thumbnails'
    },
    'reg': {
        'description': 'Windows Registry file',
        'mime_types': ['application/octet-stream'],
        'processing': 'do not parse'
    },
    'cer': {
        'description': 'Certificate file',
        'mime_types': ['application/x-x509-ca-cert'],
        'processing': 'could inspect certificate details'
    },
    'txt': {
        'description': 'Plain text file',
        'mime_types': ['text/plain'],
        'processing': 'parse or search as needed'
    },
    'xml': {
        'description': 'XML document',
        'mime_types': ['application/xml', 'text/xml'],
        'processing': 'could parse via ElementTree'
    },
}

# Human-readable size helper
def human_readable_size(size):
    for unit in ['B','KB','MB','GB','TB','PB']:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}EB"


def main():
    parser = argparse.ArgumentParser(
        description="Summarize file types in a directory with metadata lookup.",
        epilog="Example usage:\n  %(prog)s           # scan current directory\n  %(prog)s /path/to/dir --human"
    )
    parser.add_argument(
        "directory", nargs="?", default=".", help="Directory to scan (default: current directory)"
    )
    parser.add_argument(
        "--human", action="store_true", help="Show human-readable sizes (e.g., 1.23MB)"
    )
    args = parser.parse_args()

    # Validate directory parameter
    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory.")
        parser.print_help()
        sys.exit(1)

    # Notify user of start
    print(f"Scanning directory: {args.directory}")
    print("This may take a while for large trees...")
    start_time = time.time()

    counts = defaultdict(int)
    sizes = defaultdict(int)
    mimes = defaultdict(lambda: defaultdict(int))
    unknown = defaultdict(int)
    unknown_samples = {}

    mime_detector = magic.Magic(mime=True)
    file_count = 0

    for root, _, files in os.walk(args.directory):
        for name in files:
            file_count += 1
            # Print a dot for every file to indicate progress
            print('.', end='', flush=True)

            path = os.path.join(root, name)
            ext = os.path.splitext(name)[1].lower().lstrip('.')
            if not ext:
                try:
                    ext = 'drawio' if zipfile.is_zipfile(path) else 'no_extension'
                except:
                    ext = 'no_extension'

            counts[ext] += 1
            try:
                size = os.path.getsize(path)
                sizes[ext] += size
            except OSError:
                continue

            try:
                mime = mime_detector.from_file(path)
            except:
                mime = 'unknown'
            mimes[ext][mime] += 1

            if ext not in EXTENSION_METADATA:
                unknown[ext] += 1
                if ext not in unknown_samples:
                    unknown_samples[ext] = (path, mime)

    # Finish progress indicator
    print()  # newline after dots
    elapsed = time.time() - start_time
    print(f"Scan complete. Processed {file_count} files in {elapsed:.2f}s.")

    size_strs = {
        ext: (human_readable_size(sizes[ext]) if args.human else str(sizes[ext]))
        for ext in sizes
    }

    col_ext, col_cnt, col_sz, col_mime = "Extension", "Count", "Total Size", "Top MIME"
    w_ext = max(len(col_ext), *(len(e) for e in counts))
    w_cnt = max(len(col_cnt), *(len(str(c)) for c in counts.values()))
    w_sz  = max(len(col_sz), *(len(s) for s in size_strs.values()))

    header = f"{col_ext:{w_ext}}  {col_cnt:{w_cnt}}  {col_sz:{w_sz}}  {col_mime}"
    print(header)
    print("-" * len(header))
    for ext in sorted(counts):
        top_mime = max(mimes[ext].items(), key=lambda x: x[1])[0]
        print(f"{ext:{w_ext}}  {counts[ext]:{w_cnt}}  {size_strs[ext]:{w_sz}}  {top_mime}")

    print("\nMetadata for known extensions:")
    for ext, meta in sorted(EXTENSION_METADATA.items()):
        print(f"- {ext}: {meta['description']} (Typical MIME: {', '.join(meta['mime_types'])})")
        print(f"    Processing: {meta['processing']}")

    if unknown:
        print("\nUnknown file types detected:")
        for ext, cnt in unknown.items():
            sample_path, sample_mime = unknown_samples[ext]
            print(f"  * {ext}: {cnt} files (e.g. {sample_path}, MIME: {sample_mime})")

if __name__ == "__main__":
    main()

"""Expose CI test failures as annotations without requiring access to raw logs."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')


def escaped(value):
    return value.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')


path = Path(sys.argv[1])
if path.is_file():
    root = ET.parse(path).getroot()
    for case in root.iter('testcase'):
        for result in list(case.findall('failure')) + list(case.findall('error')):
            title = escaped(case.get('name', 'Test failure')).replace(',', '%2C')
            message = escaped((result.text or result.get('message', 'Test failed'))[-10000:])
            print(f'::error title={title}::{message}')

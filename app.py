"""NeuroIris desktop entry point. Use ``python -m src.cli`` for automation."""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='NeuroIris — Aluminium Vision Inspection')
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'config.yaml')
    parser.add_argument('--diagnostics', action='store_true', help='Print actual runtime capability information and exit')
    parser.add_argument('--smoke-test', action='store_true', help='Construct/render/close the real HMI; use QT_QPA_PLATFORM=offscreen on headless hosts')
    args = parser.parse_args(argv)
    from src.utils.config import load_config
    config = load_config(args.config)
    from src.service import NeuroIrisService, _jsonable
    service = NeuroIrisService(config)
    if args.diagnostics:
        print(json.dumps(_jsonable(service.diagnostics()), indent=2))
        return 0
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication
    from src.ui.neuroiris import MainWindow
    from src.utils.logging_setup import setup_logging
    setup_logging(config)
    application = QApplication(sys.argv[:1])
    application.setApplicationName('NeuroIris — Aluminium Vision Inspection')
    application.setOrganizationName('Stavatech Technology Private Limited')
    window = MainWindow(service)
    window.show()
    if args.smoke_test:
        def check():
            try:
                assert window.isVisible() and not window.grab().isNull()
                print('NeuroIris HMI constructed and rendered successfully', flush=True)
                window.close()
                application.exit(0)
            except Exception:
                logging.exception('HMI readiness failed')
                application.exit(1)
        QTimer.singleShot(400, check)
    try:
        return application.exec()
    finally:
        service.stop_camera()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        logging.exception('NeuroIris startup failed')
        print(f'NeuroIris could not start: {exc}', file=sys.stderr)
        raise SystemExit(1)

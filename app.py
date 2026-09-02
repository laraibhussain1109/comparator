from __future__ import annotations
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox

from src.ui.main_window import MainWindow
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging


def main() -> int:
    root = Path(__file__).resolve().parent
    try:
        config = load_config(root / "config.yaml", root)
        setup_logging(config)
        app = QApplication(sys.argv)
        app.setApplicationName("AI Casting Inspection System")
        window = MainWindow(config)
        window.show()
        return app.exec()
    except Exception as exc:
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Startup error", f"The inspection system could not start.\n\n{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Cross-platform automation using exactly the HMI's application service."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from src.service import NeuroIrisService, _jsonable
from src.utils.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m src.cli')
    parser.add_argument('--config', type=Path, default=Path(__file__).parents[1] / 'config.yaml')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('diagnostics')
    commands.add_parser('dataset')
    commands.add_parser('models')
    commands.add_parser('history')
    add = commands.add_parser('import')
    add.add_argument('image', type=Path)
    add.add_argument('--view', required=True)
    add.add_argument('--label', required=True, choices=['GOOD', 'NG'])
    add.add_argument('--part-id', required=True)
    add.add_argument('--session-id', default='')
    for name in ('train', 'evaluate', 'benchmark'):
        command = commands.add_parser(name)
        command.add_argument('--view', required=True, help='inner_face, ribbed_face, or all for complete-view evaluation')
        command.add_argument('--backend', choices=['geometry', 'efficientad', 'hybrid'], default='hybrid')
    inspect = commands.add_parser('inspect')
    inspect.add_argument('--part-id', required=True)
    inspect.add_argument('--inner-face', type=Path)
    inspect.add_argument('--ribbed-face', type=Path)
    inspect.add_argument('--backend', choices=['geometry', 'efficientad', 'hybrid'], default='hybrid')
    args = parser.parse_args(argv)
    service = NeuroIrisService(load_config(args.config))
    progress = lambda percent, stage, message: print(f'{percent}% {stage}: {message}', file=sys.stderr)
    if args.command == 'diagnostics':
        result = service.diagnostics()
    elif args.command == 'dataset':
        result = service.dataset_stats()
    elif args.command == 'models':
        result = service.models()
    elif args.command == 'history':
        result = service.history()
    elif args.command == 'import':
        result = service.import_sample(args.image, args.view, args.label, args.part_id, args.session_id)
    elif args.command == 'train':
        result = service.train(args.view, args.backend, progress)
    elif args.command == 'evaluate':
        result = service.evaluate(args.view, args.backend, progress)
    elif args.command == 'benchmark':
        result = service.benchmark(args.view, progress)
    elif args.command == 'inspect':
        from src.utils.image_utils import read_image
        service.set_mode(args.backend)
        service.begin_part(args.part_id)
        for view in ('inner_face', 'ribbed_face'):
            image = getattr(args, view)
            if image:
                outcome = service.inspect_frame(read_image(image), view, args.part_id)
                if outcome['verdict'] == 'FAIL':
                    break
        result = service.finish_part()
    print(json.dumps(_jsonable(result), indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)

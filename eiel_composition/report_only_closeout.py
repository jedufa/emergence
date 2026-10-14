"""Public composition seam for report publication and exact lifecycle retirement."""
from engine.campaigns.report_only_closeout import (
    closeout_report_only_run_v1,
    read_report_markdown_v1,
)

__all__ = ['closeout_report_only_run_v1', 'read_report_markdown_v1']


def main(argv=None):
    """Operator front door: exact terminal roots and an authored report."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(prog='eiel storage finish-run')
    parser.add_argument('report', type=Path)
    parser.add_argument('--run-root', type=Path, action='append', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--human', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = closeout_report_only_run_v1(
            report_path=args.report, report_markdown=read_report_markdown_v1(args.report),
            roots=args.run_root, source_commit=args.source_commit, run_id=args.run_id, apply=args.apply,
        )
    except (OSError, ValueError) as error:
        result = {'status': 'CLEANUP_PENDING', 'error': str(error)}
        print(json.dumps(result, indent=2))
        return 1
    if args.human:
        print(f"Run cleanup: {result['status']}\nReport: {result['report_path']}\n"
              f"Bytes released: {result.get('cleanup', {}).get('bytes_released', 0)}")
    else:
        print(json.dumps(result, indent=2))
    return 0

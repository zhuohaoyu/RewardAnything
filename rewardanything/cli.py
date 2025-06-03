"""Command-line interface for RewardAnything."""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RewardAnything CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start RewardAnything server')
    serve_parser.add_argument("-c", "--config", required=True, help="Path to configuration file")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--base-output-path", default="./outputs", 
                             help="Base directory for storing batch outputs")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'serve':
        # Set up arguments for serve module
        serve_args = [
            '--config', args.config,
            '--port', str(args.port),
            '--host', args.host,
            '--base-output-path', args.base_output_path
        ]
        
        # Replace sys.argv with serve arguments
        original_argv = sys.argv.copy()
        sys.argv = ['rewardanything-serve'] + serve_args
        
        try:
            from .serve import main as serve_main
            serve_main()
        finally:
            # Restore original argv
            sys.argv = original_argv
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 
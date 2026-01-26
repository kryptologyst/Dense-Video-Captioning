#!/usr/bin/env python3
"""Script to run the demo application."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run demo application")
    parser.add_argument("--port", type=int, default=8501,
                       help="Port to run the demo on")
    parser.add_argument("--host", type=str, default="localhost",
                       help="Host to run the demo on")
    
    args = parser.parse_args()
    
    demo_path = Path(__file__).parent.parent / "demo" / "app.py"
    
    if not demo_path.exists():
        print(f"Demo file not found: {demo_path}")
        sys.exit(1)
    
    print(f"Starting demo on {args.host}:{args.port}")
    print("Open your browser and navigate to the URL shown above")
    
    # Run streamlit
    subprocess.run([
        "streamlit", "run", str(demo_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
    ])


if __name__ == "__main__":
    main()

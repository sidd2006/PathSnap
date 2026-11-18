"""Serve the PathSnap web UI (Laptop 2). Standard library only.

Usage:
  python ui/serve.py --api http://<laptop-1-ip>:8000      # opens the browser at the UI
  python ui/serve.py --port 8080 --no-browser

The API address can also be changed from the UI (status at the bottom left).
"""
import argparse
import functools
import http.server
import os
import urllib.parse
import webbrowser

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def main():
    ap = argparse.ArgumentParser(description="Serve the PathSnap evaluator UI")
    ap.add_argument("--api", default=os.environ.get("PATHSNAP_API", "http://localhost:8000"),
                    help="PathSnap API address (default: $PATHSNAP_API or http://localhost:8000)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=WEB)
    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as srv:
        url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}/" \
              f"?api={urllib.parse.quote(args.api, safe=':/')}"
        print(f"PathSnap UI  ->  {url}\nAPI          ->  {args.api}\nCtrl+C to stop")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

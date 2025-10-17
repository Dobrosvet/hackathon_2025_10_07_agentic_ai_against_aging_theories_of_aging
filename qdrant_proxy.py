"""
Simple HTTP proxy server for Qdrant Web UI
Serves static files from dist-qdrant/dist and proxies API requests to Qdrant server
"""
import http.server
import socketserver
import urllib.request
import urllib.error
from pathlib import Path
import mimetypes

PORT = 8080
QDRANT_API_URL = "http://localhost:6333"
STATIC_DIR = Path(__file__).parent / "dist-qdrant" / "dist"

class QdrantProxyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        # Static file extensions
        static_extensions = ['.html', '.js', '.css', '.png', '.jpg', '.jpeg', '.svg', '.ico', '.json', '.woff', '.woff2', '.ttf', '.eot']

        # Check if this is a static file request
        path_lower = self.path.split('?')[0].lower()  # Remove query params
        is_static = any(path_lower.endswith(ext) for ext in static_extensions) or self.path == '/'

        if is_static:
            # Serve static files
            super().do_GET()
        else:
            # Everything else is an API request - proxy to Qdrant
            self.proxy_to_qdrant()

    def do_POST(self):
        self.proxy_to_qdrant()

    def do_PUT(self):
        self.proxy_to_qdrant()

    def do_DELETE(self):
        self.proxy_to_qdrant()

    def do_PATCH(self):
        self.proxy_to_qdrant()

    def proxy_to_qdrant(self):
        """Proxy request to Qdrant API server"""
        try:
            # Read request body if present
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else None

            # Build full URL
            url = f"{QDRANT_API_URL}{self.path}"

            # Create request
            headers = {
                'Content-Type': self.headers.get('Content-Type', 'application/json')
            }

            req = urllib.request.Request(url, data=body, headers=headers, method=self.command)

            # Send request to Qdrant
            with urllib.request.urlopen(req) as response:
                # Send response back to client
                self.send_response(response.status)

                # Copy headers
                for header, value in response.headers.items():
                    if header.lower() not in ['connection', 'transfer-encoding']:
                        self.send_header(header, value)

                self.end_headers()

                # Copy body
                self.wfile.write(response.read())

        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(e.read())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Proxy error: {str(e)}".encode())

if __name__ == "__main__":
    with socketserver.TCPServer(("0.0.0.0", PORT), QdrantProxyHandler) as httpd:
        print(f"Serving Qdrant Web UI with proxy at http://localhost:{PORT}")
        print(f"Proxying API requests to {QDRANT_API_URL}")
        httpd.serve_forever()

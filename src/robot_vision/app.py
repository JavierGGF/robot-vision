"""
HTTP connection test between Blockly and the vision application.

Author: Javier G. Fontanet
"""

from flask import Flask, jsonify

app = Flask(__name__)


@app.get("/health")
def health():
    """Reply without connecting to or moving the robot."""
    print("Connection test received.", flush=True)

    return jsonify(
        status="ok",
        message="Vision application is ready"
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5050,
        debug=False,
        use_reloader=False
    )
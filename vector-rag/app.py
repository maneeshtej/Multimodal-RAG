from flask import Flask, request, jsonify
import subprocess, tempfile, os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/upload", methods=["POST"])
@app.route("/upload", methods=["POST"])
def upload():
    print("\n========== /upload CALLED ==========")
    try:
        pdf = request.files.get("file")
        course_id = request.form.get("courseId")
        print("→ Received file:", pdf)
        print("→ Course ID:", course_id)

        if not pdf or not course_id:
            return jsonify({"status": "error", "message": "Missing file or courseId"}), 400

        # Always save as ./input/cloud3.pdf
        # Build path to ingest/input folder
        ingest_input_dir = os.path.join(os.getcwd(), "src", "ingest", "input")
        os.makedirs(ingest_input_dir, exist_ok=True)

        # Always save as cloud3.pdf inside ingest/input
        dest = os.path.join(ingest_input_dir, "cloud3.pdf")
        pdf.save(dest)
        print("✅ File saved at:", dest)

        # Run ingestion (no filename argument)
        ingest = subprocess.run(
            ["python3", "src/pipeline.py", "--all"],
            capture_output=True,
            text=True
        )
        print("INGEST return code:", ingest.returncode)
        print("INGEST stdout:\n", ingest.stdout)
        print("INGEST stderr:\n", ingest.stderr)

        if ingest.returncode != 0:
            return jsonify({
                "status": "error",
                "message": f"Ingest failed: {ingest.stderr.strip()}"
            }), 500

        # Run embedding
        embed = subprocess.run(
            ["python3", "src/pipeline.py", "--embed"],
            capture_output=True,
            text=True
        )
        print("EMBED return code:", embed.returncode)
        print("EMBED stdout:\n", embed.stdout)
        print("EMBED stderr:\n", embed.stderr)

        if embed.returncode != 0:
            return jsonify({
                "status": "error",
                "message": f"Embed failed: {embed.stderr.strip()}"
            }), 500

        print("✅ File processed and indexed successfully.")

        subprocess.run(
            ["python3", "src/embedd/pipeline.py", "index"]
        )

        print("File indexed")
        return jsonify({
            "status": "ok",
            "message": "File processed and indexed successfully",
            "courseId": course_id
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500





@app.route("/ask", methods=["GET"])
def ask():
    course_id = request.args.get("courseId")
    query = request.args.get("query")

    if not course_id or not query:
        return jsonify({"status": "error", "message": "Missing parameters"}), 400

    try:
        # Run the chat subcommand (same as your CLI)
        print("running query....")
        result = subprocess.run(
            ["python3", "src/embedd/pipeline.py", "chat", "--query", query],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": result.stderr.strip()
            }), 500

        return jsonify({
            "status": "ok",
            "answer": result.stdout.strip()
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500




@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

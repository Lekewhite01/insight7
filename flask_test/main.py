import json
import os
from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/service', methods=['POST'])
def service():
    data = json.loads(request.data)
    text = data.get("text")
    result = []
    for word in text:
        output = word.upper()
        result.append(output)
    if text is None:
        return jsonify({"message":"text not found"})
    else:
        return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
from flask import Flask, request, jsonify
from flask_cors import CORS

print("Initializing server")
app = Flask(__name__)
print("initializing CORS")
CORS(app)





if __name__ == "__main__":
    app.run(debug=True)






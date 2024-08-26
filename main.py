from flask import Flask, request, jsonify
from flask_cors import CORS

from engine import Engine
from utils.search.search_engine import SearchEngine


print("Initializing server")
app = Flask(__name__)
print("initializing CORS")
CORS(app)
print("Initializing engine")
engine = SearchEngine()
initial_engine = Engine()

# Initialize VideoSearchEngine
initial_engine = Engine()

# Process files in a specified directory
input_directory = "/Users/benyamindamircheli/Documents/Mirror Test" # change this to a directory with random stuff
print(f"Processing files in {input_directory}")
initial_engine.process_all_files(input_directory)

print('getting collection')
print(engine.collection.get())


@app.route('/search', methods=['POST', 'GET'])
def search():
    try:
        query = request.json['query']
        print(f"Querying {query}")
        results = engine.search(query, "text")
        return jsonify({
            'results': results
        })
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
    

if __name__ == "__main__":
    app.run(debug=True)






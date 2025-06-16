from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    # This route simply returns a plain text message.
    # It uses no models, no extra libraries.
    return "Hello World! The server is running."

# Note: We don't need the `if __name__ == '__main__':` block
# because Gunicorn handles running the app.

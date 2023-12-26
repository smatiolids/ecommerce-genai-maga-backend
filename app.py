from flask_cors import CORS
from flask import Flask
from cqlsession import getCQLSession, getCQLKeyspace
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

session = getCQLSession()
keyspace = getCQLKeyspace()

from api import chat, search

app = Flask(__name__)

app.register_blueprint(chat.bp)
app.register_blueprint(search.bp)
cors = CORS(app)
# app.register_blueprint(search.bp)

if __name__ == '__main__':
    app.run(debug=True)

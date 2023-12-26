"""
Utilities to provide connection to Astra DB (and local Cassandra)
"""

import os
from dotenv import find_dotenv, load_dotenv
from cassandra.cluster import (
    Cluster,
)
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import ordered_dict_factory

# this will climb the directory tree looking for the file
dotenv_file = find_dotenv('.env')
load_dotenv(dotenv_file, override=True)

DSE_KEYSPACE = os.environ.get('DSE_KEYSPACE')
DSE_CLUSTER = os.environ.get('DSE_CLUSTER')
DSE_PORT = os.environ.get('DSE_PORT', '9042')
DSE_USER = os.environ.get('DSE_USER')
DSE_PASSWORD = os.environ.get('DSE_PASSWORD')

def getCQLSession():
    print(f"Connecting with {DSE_CLUSTER.split(',')}.{DSE_KEYSPACE}:{DSE_PORT} with user {DSE_USER}:{DSE_PASSWORD[:4]}")
    cluster = Cluster(
        contact_points=DSE_CLUSTER.split(","),
        auth_provider=PlainTextAuthProvider(
            DSE_USER,
            DSE_PASSWORD,
        ),
        port=DSE_PORT)
    localSession = cluster.connect()
    localSession.row_factory = ordered_dict_factory
    return localSession


def getCQLKeyspace():
    return DSE_KEYSPACE

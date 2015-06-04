__author__ = 'christopher'

import json


def load_db(db_loc):
    lines = open(db_loc, 'r').readlines()
    db = []
    for line in lines:
        db.append(json.loads(line))
    return db
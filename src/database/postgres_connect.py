import configparser

import psycopg2


def connect_psql(autocommit: bool = False):
    cfg = configparser.ConfigParser()
    cfg.read("config.txt")

    connection = psycopg2.connect(user=cfg["POSTGRES"]["USER"],
                                  password=cfg["POSTGRES"]["PASSWORD"],
                                  host="127.0.0.1",
                                  port="5432",
                                  database="reddit_features")
    if autocommit:
        connection.autocommit = True
    return connection
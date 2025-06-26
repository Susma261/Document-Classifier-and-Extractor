import sqlite3

conn = sqlite3.connect('logs.db')
c = conn.cursor()

for row in c.execute("SELECT * FROM logs"):
    print(row)

conn.close()

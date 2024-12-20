class Logger:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()  # output visible immediately

    def flush(self):
        for file in self.files:
            file.flush()
import time


class Performance:
    def __init__(self, debug=True):
        self.start_time = 0
        self.debug = debug

    def tick(self):

        if self.debug:
            if self.start_time > 0:
                print('Warning you forget to call performance.end()')
            self.start_time = time.time()

    def end(self, title):
        if self.debug:
            end_time = (time.time() - self.start_time) * 1000
            self.start_time = 0
            print("{} take {} MS".format(title, end_time))

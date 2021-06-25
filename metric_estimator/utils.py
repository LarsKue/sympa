
from time import perf_counter


class Timer:
    def __init__(self, name=None, *, transform=lambda x: 1000.0 * x, unit_name="ms", precision=4, verbose=True):
        self.name = name
        self.begin = None
        self.end = None
        self.transform = transform
        self.unit_name = unit_name
        self.precision = precision
        self.verbose = verbose

    def start(self):
        self.begin = perf_counter()

    def stop(self):
        self.end = perf_counter()

    def duration(self):
        return self.transform(self.end - self.begin)

    def __enter__(self):
        self.start()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.stop()
        if self.verbose:
            print(self.message())

    def message(self):
        message = "Execution "
        if self.name is not None:
            message += f"of {self.name} "
        message += f"took {self.duration():.{self.precision}f} {self.unit_name}."
        return message

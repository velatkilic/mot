from sys import stderr
class Logger:
    ERROR=0
    QUIET=0
    BASIC=1
    WARNING=2
    DETAIL=3
    DEBUG=4

    io_level = 0

    @classmethod
    def set_io_level(cls, lev):
        cls.io_level = lev

    @classmethod
    def basic(cls, message):
        if cls.io_level >= cls.BASIC:
            print(message)

    @classmethod
    def warning(cls, message):
        if cls.io_level >= cls.WARNING:
            print("Warning! " + message)

    @classmethod
    def error(cls, message):
        print("Error! " + message, file = stderr)

    @classmethod
    def detail(cls, message):
        if cls.io_level >= cls.DETAIL:
            print(message)

    @classmethod
    def debug(cls, message):
        if cls.io_level >= cls.DEBUG:
            print(message)
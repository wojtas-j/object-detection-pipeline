# Script for exceptions

# DATASETS EXCEPTIONS
class DataUtilsError(Exception):
    pass


class InvalidInputError(Exception):
    pass


class DownloadError(Exception):
    pass


class ExtractionError(Exception):
    pass

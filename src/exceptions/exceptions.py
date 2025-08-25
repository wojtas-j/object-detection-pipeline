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


class TrainingError(Exception):
    pass


class EvaluationError(Exception):
    pass


class YamlConfigError(Exception):
    pass

class MetricsLoggingError(Exception):
    pass

class PredictionError(Exception):
    pass
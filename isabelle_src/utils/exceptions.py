class ContextTooLongError(Exception):
    pass

class PISAError(Exception):
    pass

class PISAParseError(PISAError):
    pass

class UnkRequestException(Exception):
    pass
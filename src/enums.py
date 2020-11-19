from enum import Enum, auto


class EntityType(Enum):
    SUBREDDIT = auto()
    USER = auto()


class PostType(Enum):
    POST = auto()
    COMMENT = auto()
    ALL = auto()


class DataType(Enum):
    CSV = auto()
    JSON = auto()

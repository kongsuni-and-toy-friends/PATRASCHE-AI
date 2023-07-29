try :
    from emo_classifier import *
    from ner_classifier import *
    from gd_generator import *
    from topic_classifier import *
    from yes_no_classifier import *
    from danger_detector import *
    from FER import *
except Exception :
    from .emo_classifier import *
    from .ner_classifier import *
    from .gd_generator import *
    from .topic_classifier import *
    from .yes_no_classifier import *
    from .danger_detector import *
    from .FER import *
